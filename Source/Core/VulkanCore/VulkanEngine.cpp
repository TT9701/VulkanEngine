#include "VulkanEngine.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"

#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Platform/Window.hpp"
#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanDescriptors.hpp"
#include "VulkanHelper.hpp"
#include "VulkanImage.hpp"
#include "VulkanShader.hpp"
#include "VulkanSwapchain.hpp"

VulkanEngine::VulkanEngine()
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

    mSceneUniformBuffer = mSPContext->CreateUniformBuffer(sizeof(SceneData));

    mRWBuffer = mSPContext->CreateStorageBuffer(
        sizeof(glm::vec4) * mSPWindow->GetWidth() * mSPWindow->GetHeight());

    CreateDescriptors();
    CreatePipelines();

    CreateBoxData();

#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // mFactoryModel =
    //     MakeShared<Model>("../../../Models/RM_HP_59930007DR0130HP000.fbx");

    CISDI_3DModelDataConverter converter {
        "../../../Models/RM_HP_59930007DR0130HP000.fbx"};

    // converter.Execute();

    auto cisdiModelPath = "../../../Models/RM_HP_59930007DR0130HP000.cisdi";

    auto meshes =
        CISDI_3DModelDataConverter::LoadCISDIModelData(cisdiModelPath);

    mFactoryModel = MakeShared<Model>(meshes);

    mFactoryModel->GenerateBuffers(mSPContext.get(), this);
}

VulkanEngine::~VulkanEngine() {
    mSPContext->GetDeviceHandle().waitIdle();
}

void VulkanEngine::Run() {
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

void VulkanEngine::Draw() {
    auto swapchainImage =
        mSPSwapchain->GetImageHandle(mSPSwapchain->AcquireNextImageIndex());

    const uint64_t graphicsFinished =
        mSPContext->GetTimelineSemphore()->GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished     = graphicsFinished + 2;

    // Compute Draw
    {
        auto cmd = mSPCmdManager->GetCmdBufferToBegin();

        mDrawImage->TransitionLayout(cmd.GetHandle(),
                                     vk::ImageLayout::eGeneral);

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

        mDrawImage->TransitionLayout(cmd.GetHandle(),
                                     vk::ImageLayout::eColorAttachmentOptimal);
        mDepthImage->TransitionLayout(cmd.GetHandle(),
                                      vk::ImageLayout::eDepthAttachmentOptimal);

        DrawMesh(cmd.GetHandle());

        mDrawImage->TransitionLayout(cmd.GetHandle(),
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

UniquePtr<SDLWindow> VulkanEngine::CreateSDLWindow() {
    return MakeUnique<SDLWindow>();
}

UniquePtr<VulkanContext> VulkanEngine::CreateContext() {
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

    VulkanContext::EnableDefaultFeatures();

    return MakeUnique<VulkanContext>(
        mSPWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

UniquePtr<VulkanSwapchain> VulkanEngine::CreateSwapchain() {
    return MakeUnique<VulkanSwapchain>(
        mSPContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mSPWindow->GetWidth()),
                      static_cast<uint32_t>(mSPWindow->GetHeight())});
}

SharedPtr<VulkanImage> VulkanEngine::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    return mSPContext->CreateImage2D(
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, drawImageExtent,
        vk::Format::eR16G16B16A16Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

SharedPtr<VulkanImage> VulkanEngine::CreateDepthImage() {
    vk::Extent3D depthImageExtent {
        static_cast<uint32_t>(mSPWindow->GetWidth()),
        static_cast<uint32_t>(mSPWindow->GetHeight()), 1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    return mSPContext->CreateImage2D(VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                     depthImageExtent, vk::Format::eD32Sfloat,
                                     depthImageUsage,
                                     vk::ImageAspectFlagBits::eDepth);
}

UniquePtr<ImmediateSubmitManager> VulkanEngine::CreateImmediateSubmitManager() {
    return MakeUnique<ImmediateSubmitManager>(
        mSPContext.get(),
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

UniquePtr<VulkanCommandManager> VulkanEngine::CreateCommandManager() {
    return MakeUnique<VulkanCommandManager>(
        mSPContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

void VulkanEngine::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
}

void VulkanEngine::CreateDescriptors() {

    CreateBackgroundComputeDescriptors();
    CreateMeshDescriptors();
    CreateDrawQuadDescriptors();
}

void VulkanEngine::CreateBoxData() {
    ::std::array<Vertex, 8> vertices {};

    vertices[0].position = {-1, -1, 1, 0};
    vertices[1].position = {1, -1, 1, 0};
    vertices[2].position = {-1, 1, 1, 0};
    vertices[3].position = {1, 1, 1, 0};
    vertices[4].position = {-1, -1, -1, 0};
    vertices[5].position = {1, -1, -1, 0};
    vertices[6].position = {-1, 1, -1, 0};
    vertices[7].position = {1, 1, -1, 0};

    vertices[0].texcoords = {0.0f, 0.0f};
    vertices[1].texcoords = {1.0f, 0.0f};
    vertices[2].texcoords = {0.0f, 1.0f};
    vertices[3].texcoords = {1.0f, 1.0f};
    vertices[4].texcoords = {0.0f, 0.0f};
    vertices[5].texcoords = {1.0f, 0.0f};
    vertices[6].texcoords = {0.0f, 1.0f};
    vertices[7].texcoords = {1.0f, 1.0f};

    ::std::array<uint32_t, 36> indices {//Top
                                        2, 7, 6, 2, 3, 7,
                                        //Bottom
                                        0, 4, 5, 0, 5, 1,
                                        //Left
                                        0, 2, 6, 0, 6, 4,
                                        //Right
                                        1, 7, 3, 1, 5, 7,
                                        //Front
                                        0, 3, 2, 0, 1, 3,
                                        //Back
                                        4, 6, 7, 4, 7, 5};

    mBoxMesh = UploadMeshData(indices, vertices);
}

SharedPtr<VulkanImage> VulkanEngine::CreateErrorCheckTexture() {
    uint32_t black   = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    return mSPContext->CreateImage2D(
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, VkExtent3D {16, 16, 1},
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor, pixels.data(), this);
}

UniquePtr<VulkanDescriptorManager> VulkanEngine::CreateDescriptorManager() {
    std::vector<DescPoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};

    return MakeUnique<VulkanDescriptorManager>(mSPContext.get(), 10, sizes);
}

UniquePtr<VulkanPipelineManager> VulkanEngine::CreatePipelineManager() {
    return MakeUnique<VulkanPipelineManager>(mSPContext.get());
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> VulkanEngine::CreateExternalImage() {
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

void VulkanEngine::CreateExternalTriangleData() {
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

void VulkanEngine::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());
    mCUDASignalSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void VulkanEngine::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mSPContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

GPUMeshBuffers VulkanEngine::UploadMeshData(std::span<uint32_t> indices,
                                            std::span<Vertex>   vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(vertices[0]);
    const size_t indexBufferSize  = indices.size() * sizeof(indices[0]);

    GPUMeshBuffers newMesh {};
    newMesh.mVertexBuffer = mSPContext->CreatePersistentBuffer(
        vertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer
                              | vk::BufferUsageFlagBits::eTransferDst
                              | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    newMesh.mIndexBuffer = mSPContext->CreatePersistentBuffer(
        indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer
                             | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(newMesh.mVertexBuffer->GetHandle());

    newMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    auto staging =
        mSPContext->CreateStagingBuffer(vertexBufferSize + indexBufferSize);

    void* data = staging->GetAllocationInfo().pMappedData;
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    mSPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), newMesh.mVertexBuffer->GetHandle(),
                       vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), newMesh.mIndexBuffer->GetHandle(),
                       indexCopy);
    });

    return newMesh;
}

void VulkanEngine::UpdateScene() {
    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(mSPWindow->GetWidth())
                             / static_cast<float>(mSPWindow->GetHeight()),
                         10000.0f, 0.01f);

    proj[1][1] *= -1;

    mSceneData.cameraPos = glm::vec4 {mMainCamera.mPosition, 1.0f};
    mSceneData.view      = view;
    mSceneData.proj      = proj;
    mSceneData.viewProj  = proj * view;
    UpdateSceneUBO();
}

void VulkanEngine::UpdateSceneUBO() {
    memcpy(mSceneUniformBuffer->GetAllocationInfo().pMappedData, &mSceneData,
           sizeof(mSceneData));
}

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eStorageImage);
    mDescriptorManager->AddDescSetLayoutBinding(
        1, 1, vk::DescriptorType::eStorageBuffer);

    const auto drawImageSetLayout = mDescriptorManager->BuildDescSetLayout(
        "DrawImage_Layout_0", vk::ShaderStageFlagBits::eCompute);

    const auto drawImageDesc =
        mDescriptorManager->Allocate("DrawImage_Desc_0", drawImageSetLayout);

    mDescriptorManager->WriteImage(0,
                                   {VK_NULL_HANDLE, mDrawImage->GetViewHandle(),
                                    vk::ImageLayout::eGeneral},
                                   vk::DescriptorType::eStorageImage);

    mDescriptorManager->WriteBuffer(
        1,
        {mRWBuffer->GetHandle(), 0,
         sizeof(glm::vec4) * mSPWindow->GetWidth() * mSPWindow->GetHeight()},
        vk::DescriptorType::eStorageBuffer);

    mDescriptorManager->UpdateSet(drawImageDesc);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void VulkanEngine::CreateBackgroundComputePipeline() {
    ::std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("DrawImage_Layout_0")};

    auto backgroundPipelineLayout = mPipelineManager->CreateLayout(
        "BackgoundCompute_Layout", setLayouts, {});

    VulkanShader computeDrawShader {mSPContext.get(), "computeDraw",
                                    "../../Shaders/BackGround.comp.spv",
                                    ShaderStage::Compute};

    auto& builder = mPipelineManager->GetComputePipelineBuilder();

    auto backgroundComputePipeline =
        builder.SetShader(computeDrawShader)
            .SetLayout(backgroundPipelineLayout->GetHandle())
            .Build("BackgroundCompute_Pipeline");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void VulkanEngine::CreateMeshPipeline() {
    std::vector<VulkanShader> shaders;
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
        .SetColorAttachmentFormat(mDrawImage->GetFormat())
        .SetDepthStencilFormat(mDepthImage->GetFormat())
        .Build("TriangleDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void VulkanEngine::CreateMeshDescriptors() {
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

    mDescriptorManager->WriteImage(
        0,
        {mSPContext->GetDefaultNearestSamplerHandle(),
         mErrorCheckImage->GetViewHandle(),
         vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(triangleDesc1);
}

void VulkanEngine::CreateDrawQuadDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto quadSetLayout = mDescriptorManager->BuildDescSetLayout(
        "Quad_Layout_0", vk::ShaderStageFlagBits::eFragment);

    const auto quadDesc =
        mDescriptorManager->Allocate("Quad_Desc_0", quadSetLayout);

    mDescriptorManager->WriteImage(
        0,
        {mSPContext->GetDefaultLinearSamplerHandle(),
         mDrawImage->GetViewHandle(), vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(quadDesc);
}

void VulkanEngine::CreateDrawQuadPipeline() {
    std::vector<VulkanShader> shaders;
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

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));

    clearValue = {flash, flash, flash, 1.0f};

    auto subresource =
        Utils::GetDefaultImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

    cmd.clearColorImage(mDrawImage->GetHandle(), vk::ImageLayout::eGeneral,
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

        cmd.dispatch(::std::ceil(mDrawImage->GetExtent3D().width / 16.0),
                     ::std::ceil(mDrawImage->GetExtent3D().height / 16.0), 1);
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

void VulkanEngine::DrawMesh(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mDrawImage->GetViewHandle())
        .setImageLayout(mDrawImage->GetLayout())
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment.setImageView(mDepthImage->GetViewHandle())
        .setImageLayout(mDepthImage->GetLayout())
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {{0, 0},
                                   {mDrawImage->GetExtent3D().width,
                                    mDrawImage->GetExtent3D().height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("TriangleDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport.setWidth(mDrawImage->GetExtent3D().width)
        .setHeight(mDrawImage->GetExtent3D().height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mDrawImage->GetExtent3D().width, mDrawImage->GetExtent3D().height});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("Triangle_Layout"), 0,
        {mDescriptorManager->GetDescriptor("Triangle_Desc_0"),
         mDescriptorManager->GetDescriptor("Triangle_Desc_1")},
        {});

    mDescriptorManager->WriteBuffer(
        0, {mSceneUniformBuffer->GetHandle(), 0, sizeof(SceneData)},
        vk::DescriptorType::eUniformBuffer);

    mDescriptorManager->UpdateSet(
        mDescriptorManager->GetDescriptor("Triangle_Desc_0"));

    // PushConstants pushConstants {};
    // pushConstants.mVertexBufferAddress = mBoxMesh.mVertexBufferAddress;
    // // mTriangleExternalMesh.mVertexBufferAddress;
    // pushConstants.mModelMatrix = glm::mat4(1.0f);
    // cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
    //                   vk::ShaderStageFlagBits::eVertex, 0,
    //                   sizeof(PushConstants), &pushConstants);
    //
    // cmd.bindIndexBuffer(mBoxMesh.mIndexBuffer->GetHandle(), 0,
    //                     vk::IndexType::eUint32);
    //
    // cmd.drawIndexed(36, 1, 0, 0, 0);

    cmd.bindIndexBuffer(
        mFactoryModel->GetMeshBuffer().mIndexBuffer->GetHandle(), 0,
        vk::IndexType::eUint32);

    glm::mat4 model {1.0f};
    model = glm::scale(model, glm::vec3 {0.0001f});

    auto pushContants         = mFactoryModel->GetPushContants();
    pushContants.mModelMatrix = model;

    cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
                      vk::ShaderStageFlagBits::eVertex, 0, sizeof(pushContants),
                      &pushContants);

    // for (uint32_t i = 0; i < mFactoryModel->GetMeshes().size(); ++i) {
    //     auto const& mesh = mFactoryModel->GetMeshes()[i];
    //
    //     cmd.drawIndexed(static_cast<uint32_t>(mesh.mIndices.size()), 1,
    //                     mFactoryModel->GetIndexOffsets()[i],
    //                     mFactoryModel->GetVertexOffsets()[i], 0);
    // }

    cmd.drawIndexedIndirect(mFactoryModel->GetIndirectCmdBuffer()->GetHandle(),
                            0, mFactoryModel->GetMeshes().size(),
                            sizeof(vk::DrawIndexedIndirectCommand));

    cmd.endRendering();
}

void VulkanEngine::DrawQuad(vk::CommandBuffer cmd) {
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
