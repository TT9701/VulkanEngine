#include "EngineCore.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"

#include "Context.hpp"
#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Platform/Window.hpp"
#include "RenderResource.h"
#include "Shader.hpp"
#include "Swapchain.hpp"
#include "Descriptors.hpp"
#include "VulkanHelper.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

EngineCore::EngineCore()
    : mSPWindow(CreateSDLWindow()),
      mSPContext(CreateContext()),
      mSPSwapchain(CreateSwapchain()),
      mDrawImage(CreateDrawImage()),
      mDepthImage(CreateDepthImage()),
#ifdef CUDA_VULKAN_INTEROP
      mCUDAExternalImage(CreateExternalImage()),
#endif
      mSPCmdManager(CreateCommandManager()),
      mSPImmediateSubmitManager(CreateImmediateSubmitManager()),
      mErrorCheckImage(CreateErrorCheckTexture()),
      mDescriptorManager(CreateDescriptorManager()),
      mPipelineManager(CreatePipelineManager()) {

    mSceneUniformBuffer = mSPContext->CreateStagingBuffer(
        sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);

    mRWBuffer = mSPContext->CreateStorageBuffer(
        sizeof(glm::vec4) * mSPWindow->GetWidth() * mSPWindow->GetHeight());

    CreateDescriptors();
    CreatePipelines();

#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // mFactoryModel =
    //     MakeShared<Model>("../../../Models/RM_HP_59930007DR0130HP000.fbx");
    //
    // CISDI_3DModelDataConverter converter {
    //     "../../../Models/RM_HP_59930007DR0130HP000.fbx"};
    //
    // converter.Execute();

    auto cisdiModelPath = "../../../Models/RM_HP_59930007DR0130HP000.cisdi";

    auto meshes =
        CISDI_3DModelDataConverter::LoadCISDIModelData(cisdiModelPath);

    mFactoryModel = MakeShared<Model>(meshes);

    mFactoryModel->GenerateBuffers(mSPContext.get(), this);
}

EngineCore::~EngineCore() {
    mSPContext->GetDeviceHandle().waitIdle();
}

void EngineCore::Run() {
    bool bQuit = false;

    while (!bQuit) {
        mSPWindow->PollEvents(bQuit, mStopRendering, [&](SDL_Event* e) {
            mMainCamera.ProcessSDLEvent(e, 0.001f);
        });

        if (mStopRendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            UpdateScene();
            Draw();
        }
    }
}

void EngineCore::Draw() {
    auto swapchainImage =
        mSPSwapchain->GetImageHandle(mSPSwapchain->AcquireNextImageIndex());

    const uint64_t graphicsFinished =
        mSPContext->GetTimelineSemphore()->GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    // Compute Draw
    {
        auto cmd = mSPCmdManager->GetCmdBufferToBegin();

        Utils::TransitionImageLayout(
            cmd.GetHandle(), mDrawImage->GetTexHandle(),
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        DrawBackground(cmd.GetHandle());

        cmd.End();

        ::std::vector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             mSPSwapchain->GetReady4RenderSemHandle(), 0ui64},
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             mSPContext->GetTimelineSemaphoreHandle(), graphicsFinished}};

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mSPContext->GetTimelineSemaphoreHandle(), computeFinished}};

        mSPCmdManager->Submit(cmd.GetHandle(),
                              mSPContext->GetDevice()->GetGraphicQueue(), waits,
                              signals);
    }

    // Graphics Draw
    {
        auto cmd = mSPCmdManager->GetCmdBufferToBegin();

        Utils::TransitionImageLayout(cmd.GetHandle(),
                                     mDrawImage->GetTexHandle(),
                                     vk::ImageLayout::eGeneral,
                                     vk::ImageLayout::eColorAttachmentOptimal);

        Utils::TransitionImageLayout(cmd.GetHandle(),
                                     mDepthImage->GetTexHandle(),
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eDepthAttachmentOptimal);

        DrawMesh(cmd.GetHandle());

        Utils::TransitionImageLayout(cmd.GetHandle(),
                                     mDrawImage->GetTexHandle(),
                                     vk::ImageLayout::eColorAttachmentOptimal,
                                     vk::ImageLayout::eShaderReadOnlyOptimal);

        Utils::TransitionImageLayout(cmd.GetHandle(), swapchainImage,
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eColorAttachmentOptimal);

        DrawQuad(cmd.GetHandle());

        Utils::TransitionImageLayout(cmd.GetHandle(), swapchainImage,
                                     vk::ImageLayout::eColorAttachmentOptimal,
                                     vk::ImageLayout::ePresentSrcKHR);

        cmd.End();

        ::std::vector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eComputeShader,
             mSPContext->GetTimelineSemaphoreHandle(), computeFinished}};

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mSPContext->GetTimelineSemaphoreHandle(), allFinished},
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mSPSwapchain->GetReady4PresentSemHandle()}};

        mSPCmdManager->Submit(cmd.GetHandle(),
                              mSPContext->GetDevice()->GetGraphicQueue(), waits,
                              signals);
    }

    {
        auto cmd = mSPCmdManager->GetCmdBufferToBegin();
        cmd.End();

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mSPContext->GetTimelineSemaphoreHandle(), allFinished + 1}};

        mSPCmdManager->Submit(cmd.GetHandle(),
                              mSPContext->GetDevice()->GetGraphicQueue(), {},
                              signals);
    }

    // #ifdef CUDA_VULKAN_INTEROP
    //     waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
    //         vk::PipelineStageFlagBits2::eAllCommands,
    //         mCUDASignalSemaphore->GetVkSemaphore()));
    // #endif
    //
    // #ifdef CUDA_VULKAN_INTEROP
    //     signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
    //         vk::PipelineStageFlagBits2::eAllCommands,
    //         mCUDAWaitSemaphore->GetVkSemaphore()));
    // #endif

    mSPSwapchain->Present(mSPContext->GetDevice()->GetGraphicQueue());

    // #ifdef CUDA_VULKAN_INTEROP
    //     cudaExternalSemaphoreWaitParams waitParams {};
    //     auto cudaWait = mCUDAWaitSemaphore->GetCUDAExternalSemaphore();
    //     mCUDAStream.WaitExternalSemaphoresAsync(&cudaWait, &waitParams, 1);
    //
    //     CUDA::SimPoint(mTriangleExternalMesh.mVertexBuffer
    //                        ->GetMappedPointer(0, 3 * sizeof(Vertex))
    //                        .GetPtr(),
    //                    mFrameNum, mCUDAStream.GetHandle());
    //
    //     CUDA::SimSurface(*mCUDAExternalImage->GetSurfaceObjectPtr(), mFrameNum,
    //                      mCUDAStream.GetHandle());
    //
    //     cudaExternalSemaphoreSignalParams signalParams {};
    //     auto cudaSignal = mCUDASignalSemaphore->GetCUDAExternalSemaphore();
    //     mCUDAStream.SignalExternalSemaphoresAsyn(&cudaSignal, &signalParams, 1);
    // #endif

    ++mFrameNum;
}

UniquePtr<SDLWindow> EngineCore::CreateSDLWindow() {
    return MakeUnique<SDLWindow>();
}

UniquePtr<Context> EngineCore::CreateContext() {
    ::std::vector<::std::string> requestedInstanceLayers {};
#ifndef NDEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    auto sdlRequestedInstanceExtensions =
        mSPWindow->GetVulkanInstanceExtension();
    ::std::vector<::std::string> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifndef NDEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    ::std::vector<::std::string> enabledDeivceExtensions {};

    enabledDeivceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);

    Context::EnableDefaultFeatures();

    return MakeUnique<Context>(
        mSPWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

UniquePtr<Swapchain> EngineCore::CreateSwapchain() {
    return MakeUnique<Swapchain>(
        mSPContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mSPWindow->GetWidth()),
                      static_cast<uint32_t>(mSPWindow->GetHeight())});
}

SharedPtr<RenderResource> EngineCore::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto ptr = mSPContext->CreateTexture2D(
        drawImageExtent, vk::Format::eR16G16B16A16Sfloat, drawImageUsage);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
    return ptr;
}

SharedPtr<RenderResource> EngineCore::CreateDepthImage() {
    vk::Extent3D depthImageExtent {
        static_cast<uint32_t>(mSPWindow->GetWidth()),
        static_cast<uint32_t>(mSPWindow->GetHeight()), 1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto ptr = mSPContext->CreateTexture2D(
        depthImageExtent, vk::Format::eD32Sfloat, depthImageUsage);
    ptr->CreateTexView("Depth-Whole", vk::ImageAspectFlagBits::eDepth);

    return ptr;
}

UniquePtr<ImmediateSubmitManager> EngineCore::CreateImmediateSubmitManager() {
    return MakeUnique<ImmediateSubmitManager>(
        mSPContext.get(),
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

UniquePtr<CommandManager> EngineCore::CreateCommandManager() {
    return MakeUnique<CommandManager>(
        mSPContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

void EngineCore::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
}

void EngineCore::CreateDescriptors() {

    CreateBackgroundComputeDescriptors();
    CreateMeshDescriptors();
    CreateDrawQuadDescriptors();
}

SharedPtr<RenderResource> EngineCore::CreateErrorCheckTexture() {
    auto extent = VkExtent3D {16, 16, 1};
    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    auto ptr =
        mSPContext->CreateTexture2D(extent, vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageUsageFlagBits::eSampled
                                        | vk::ImageUsageFlagBits::eTransferDst);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

    {
        size_t dataSize = extent.width * extent.height * 4;

        auto uploadBuffer = mSPContext->CreateStagingBuffer(dataSize);
        memcpy(uploadBuffer->GetBufferMappedPtr(), pixels.data(), dataSize);

        mSPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
            Utils::TransitionImageLayout(cmd, ptr->GetTexHandle(),
                                         vk::ImageLayout::eUndefined,
                                         vk::ImageLayout::eTransferDstOptimal);

            vk::BufferImageCopy copyRegion {};
            copyRegion
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(extent);

            cmd.copyBufferToImage(
                uploadBuffer->GetBufferHandle(), ptr->GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal, copyRegion);

            Utils::TransitionImageLayout(
                cmd, ptr->GetTexHandle(), vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        });
    }

    return ptr;
}

UniquePtr<DescriptorManager> EngineCore::CreateDescriptorManager() {
    std::vector<DescPoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};

    return MakeUnique<DescriptorManager>(mSPContext.get(), 10, sizes);
}

UniquePtr<PipelineManager> EngineCore::CreatePipelineManager() {
    return MakeUnique<PipelineManager>(mSPContext.get());
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> EngineCore::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return mSPContext->CreateExternalImage2D(
        drawImageExtent, vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::CreateExternalTriangleData() {
    mTriangleExternalMesh.mVertexBuffer =
        mSPContext->CreateExternalPersistentBuffer(
            3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mTriangleExternalMesh.mIndexBuffer =
        mSPContext->CreateExternalPersistentBuffer(
            3 * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);
}

void EngineCore::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());
    mCUDASignalSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void EngineCore::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mSPContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

void EngineCore::UpdateScene() {
    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(mSPWindow->GetWidth())
                             / static_cast<float>(mSPWindow->GetHeight()),
                         10000.0f, 0.01f);

    proj[1][1] *= -1;

    mSceneData.cameraPos = glm::vec4 {mMainCamera.mPosition, 1.0f};
    mSceneData.view = view;
    mSceneData.proj = proj;
    mSceneData.viewProj = proj * view;
    UpdateSceneUBO();
}

void EngineCore::UpdateSceneUBO() {
    memcpy(mSceneUniformBuffer->GetBufferMappedPtr(), &mSceneData,
           sizeof(mSceneData));
}

void EngineCore::CreateBackgroundComputeDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eStorageImage);
    mDescriptorManager->AddDescSetLayoutBinding(
        1, 1, vk::DescriptorType::eStorageBuffer);

    const auto drawImageSetLayout = mDescriptorManager->BuildDescSetLayout(
        "DrawImage_Layout_0", vk::ShaderStageFlagBits::eCompute);

    const auto drawImageDesc =
        mDescriptorManager->Allocate("DrawImage_Desc_0", drawImageSetLayout);

    mDescriptorManager->WriteImage(
        0,
        {VK_NULL_HANDLE, mDrawImage->GetTexViewHandle("Color-Whole"),
         vk::ImageLayout::eGeneral},
        vk::DescriptorType::eStorageImage);

    mDescriptorManager->WriteBuffer(
        1,
        {mRWBuffer->GetBufferHandle(), 0,
         sizeof(glm::vec4) * mSPWindow->GetWidth() * mSPWindow->GetHeight()},
        vk::DescriptorType::eStorageBuffer);

    mDescriptorManager->UpdateSet(drawImageDesc);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void EngineCore::CreateBackgroundComputePipeline() {
    ::std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("DrawImage_Layout_0")};

    auto backgroundPipelineLayout = mPipelineManager->CreateLayout(
        "BackgoundCompute_Layout", setLayouts, {});

    Shader computeDrawShader {mSPContext.get(), "computeDraw",
                                    "../../Shaders/BackGround.comp.spv",
                                    ShaderStage::Compute};

    auto& builder = mPipelineManager->GetComputePipelineBuilder();

    auto backgroundComputePipeline =
        builder.SetShader(computeDrawShader)
            .SetLayout(backgroundPipelineLayout->GetHandle())
            .Build("BackgroundCompute_Pipeline");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void EngineCore::CreateMeshPipeline() {
    std::vector<Shader> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mSPContext.get(), "vertex",
                         "../../Shaders/Triangle.vert.spv",
                         ShaderStage::Vertex);

    shaders.emplace_back(mSPContext.get(), "fragment",
                         "../../Shaders/Triangle.frag.spv",
                         ShaderStage::Fragment);

    vk::PushConstantRange pushConstant {};
    pushConstant.setSize(sizeof(PushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("Triangle_Layout_0"),
        mDescriptorManager->GetDescSetLayout("Triangle_Layout_1")};

    auto trianglePipelineLayout = mPipelineManager->CreateLayout(
        "Triangle_Layout", setLayouts, pushConstant);

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(trianglePipelineLayout->GetHandle())
        .SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eFront,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(mDrawImage->GetTexFormat())
        .SetDepthStencilFormat(mDepthImage->GetTexFormat())
        .Build("TriangleDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void EngineCore::CreateMeshDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eUniformBuffer);

    const auto triangleSetLayout0 = mDescriptorManager->BuildDescSetLayout(
        "Triangle_Layout_0",
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto triangleSetLayout1 = mDescriptorManager->BuildDescSetLayout(
        "Triangle_Layout_1", vk::ShaderStageFlagBits::eFragment);

    const auto triangleDesc0 =
        mDescriptorManager->Allocate("Triangle_Desc_0", triangleSetLayout0);

    const auto triangleDesc1 =
        mDescriptorManager->Allocate("Triangle_Desc_1", triangleSetLayout1);

    // mDescriptorManager->WriteImage(
    //     0,
    //     {mSPContext->GetDefaultNearestSamplerHandle(),
    //      mErrorCheckImage->GetTexViewHandle("Color-Whole"),
    //      vk::ImageLayout::eShaderReadOnlyOptimal},
    //     vk::DescriptorType::eCombinedImageSampler);
    //
    // mDescriptorManager->UpdateSet(triangleDesc1);
}

void EngineCore::CreateDrawQuadDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto quadSetLayout = mDescriptorManager->BuildDescSetLayout(
        "Quad_Layout_0", vk::ShaderStageFlagBits::eFragment);

    const auto quadDesc =
        mDescriptorManager->Allocate("Quad_Desc_0", quadSetLayout);

    mDescriptorManager->WriteImage(0,
                                   {mSPContext->GetDefaultLinearSamplerHandle(),
                                    mDrawImage->GetTexViewHandle("Color-Whole"),
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
                                   vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(quadDesc);
}

void EngineCore::CreateDrawQuadPipeline() {
    std::vector<Shader> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mSPContext.get(), "vertex",
                         "../../Shaders/Quad.vert.spv", ShaderStage::Vertex);

    shaders.emplace_back(mSPContext.get(), "fragment",
                         "../../Shaders/Quad.frag.spv", ShaderStage::Fragment);

    std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("Quad_Layout_0")};

    auto quadPipelineLayout =
        mPipelineManager->CreateLayout("Quad_Layout", setLayouts);

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(quadPipelineLayout->GetHandle())
        .SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::False, vk::False)
        .SetColorAttachmentFormat(mSPSwapchain->GetFormat())
        .SetDepthStencilFormat(vk::Format::eUndefined)
        .Build("QuadDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Quad Graphics Pipeline Created");
}

void EngineCore::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));

    clearValue = {flash, flash, flash, 1.0f};

    auto subresource = vk::ImageSubresourceRange {
        vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0,
        vk::RemainingArrayLayers};

    cmd.clearColorImage(mDrawImage->GetTexHandle(), vk::ImageLayout::eGeneral,
                        clearValue, subresource);

    // Compute Draw
    {
        cmd.bindPipeline(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetComputePipeline("BackgroundCompute_Pipeline"));

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetLayoutHandle("BackgoundCompute_Layout"), 0,
            mDescriptorManager->GetDescriptor("DrawImage_Desc_0"), {});

        cmd.dispatch(::std::ceil(mDrawImage->GetTexWidth() / 16.0),
                     ::std::ceil(mDrawImage->GetTexHeight() / 16.0), 1);
    }

    // CUDA Draw
    // {
    //     auto layout = mDrawImage->GetLayout();
    //
    //     mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    //     Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
    //                                  vk::ImageLayout::eUndefined,
    //                                  vk::ImageLayout::eTransferSrcOptimal);
    //
    //     vk::ImageBlit2 blitRegion {};
    //     blitRegion
    //         .setSrcOffsets(
    //             {vk::Offset3D {},
    //              vk::Offset3D {static_cast<int32_t>(
    //                                mCUDAExternalImage->GetExtent3D().width),
    //                            static_cast<int32_t>(
    //                                mCUDAExternalImage->GetExtent3D().height),
    //                            1}})
    //         .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
    //         .setDstOffsets(
    //             {vk::Offset3D {},
    //              vk::Offset3D {
    //                  static_cast<int32_t>(mDrawImage->GetExtent3D().width),
    //                  static_cast<int32_t>(mDrawImage->GetExtent3D().height),
    //                  1}})
    //         .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    //
    //     vk::BlitImageInfo2 blitInfo {};
    //     blitInfo.setDstImage(mDrawImage->GetHandle())
    //         .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
    //         .setSrcImage(mCUDAExternalImage->GetVkImage())
    //         .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
    //         .setFilter(vk::Filter::eLinear)
    //         .setRegions(blitRegion);
    //
    //     cmd.blitImage2(blitInfo);
    //
    //     mDrawImage->TransitionLayout(cmd, layout);
    //     Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
    //                                  vk::ImageLayout::eTransferSrcOptimal,
    //                                  vk::ImageLayout::eGeneral);
    // }
}

void EngineCore::DrawMesh(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mDrawImage->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment.setImageView(mDepthImage->GetTexViewHandle("Depth-Whole"))
        .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {
            {0, 0}, {mDrawImage->GetTexWidth(), mDrawImage->GetTexHeight()}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("TriangleDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport.setWidth(mDrawImage->GetTexWidth())
        .setHeight(mDrawImage->GetTexHeight())
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent({mDrawImage->GetTexWidth(), mDrawImage->GetTexHeight()});
    cmd.setScissor(0, scissor);

    mDescriptorManager->WriteImage(
        0,
        {mSPContext->GetDefaultNearestSamplerHandle(),
         mErrorCheckImage->GetTexViewHandle("Color-Whole"),
         vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(
        mDescriptorManager->GetDescriptor("Triangle_Desc_1"));

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("Triangle_Layout"), 0,
        {mDescriptorManager->GetDescriptor("Triangle_Desc_0"),
         mDescriptorManager->GetDescriptor("Triangle_Desc_1")},
        {});

    mDescriptorManager->WriteBuffer(
        0, {mSceneUniformBuffer->GetBufferHandle(), 0, sizeof(SceneData)},
        vk::DescriptorType::eUniformBuffer);

    mDescriptorManager->UpdateSet(
        mDescriptorManager->GetDescriptor("Triangle_Desc_0"));

    cmd.bindIndexBuffer(
        mFactoryModel->GetMeshBuffer().mIndexBuffer->GetBufferHandle(), 0,
        vk::IndexType::eUint32);

    glm::mat4 model {1.0f};
    model = glm::scale(model, glm::vec3 {0.0001f});

    auto pushContants = mFactoryModel->GetPushContants();
    pushContants.mModelMatrix = model;

    cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
                      vk::ShaderStageFlagBits::eVertex, 0, sizeof(pushContants),
                      &pushContants);

    cmd.drawIndexedIndirect(
        mFactoryModel->GetIndirectCmdBuffer()->GetBufferHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawIndexedIndirectCommand));

    cmd.endRendering();
}

void EngineCore::DrawQuad(vk::CommandBuffer cmd) {
    auto imageIndex = mSPSwapchain->GetCurrentImageIndex();
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mSPSwapchain->GetImageViewHandle(imageIndex))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {{0, 0},
                                   {mSPSwapchain->GetExtent2D().width,
                                    mSPSwapchain->GetExtent2D().height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("QuadDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport.setWidth(mSPSwapchain->GetExtent2D().width)
        .setHeight(mSPSwapchain->GetExtent2D().height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent({mSPSwapchain->GetExtent2D().width,
                       mSPSwapchain->GetExtent2D().height});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           mPipelineManager->GetLayoutHandle("Quad_Layout"), 0,
                           mDescriptorManager->GetDescriptor("Quad_Desc_0"),
                           {});

    cmd.draw(3, 1, 0, 0);

    cmd.endRendering();
}

}  // namespace IntelliDesign_NS::Vulkan::Core