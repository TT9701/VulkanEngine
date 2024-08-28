#include "EngineCore.hpp"

#include "Buffer.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"

#include "Context.hpp"
#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Platform/Window.hpp"
#include "Descriptors.hpp"
#include "RenderResource.hpp"
#include "RenderResourceManager.hpp"
#include "Shader.hpp"
#include "Swapchain.hpp"
#include "VulkanHelper.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

EngineCore::EngineCore()
    : mPWindow(CreateSDLWindow()),
      mPContext(CreateContext()),
      mPSwapchain(CreateSwapchain()),
      mRenderResManager(CreateRenderResourceManager()),
#ifdef CUDA_VULKAN_INTEROP
      mCUDAExternalImage(CreateExternalImage()),
#endif
      mPCmdManager(CreateCommandManager()),
      mPImmediateSubmitManager(CreateImmediateSubmitManager()),
      mDescriptorManager(CreateDescriptorManager()),
      mPipelineManager(CreatePipelineManager()) {
    CreateDrawImage();
    CreateDepthImage();
    CreateErrorCheckTexture();

    mRenderResManager->CreateBuffer("SceneUniformBuffer", sizeof(SceneData),
                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                    Buffer::MemoryType::Staging);

    mRenderResManager->CreateBuffer(
        "RWBuffer",
        sizeof(glm::vec4) * mPWindow->GetWidth() * mPWindow->GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer,
        Buffer::MemoryType::DeviceLocal);

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // mFactoryModel =
    //     MakeShared<Model>(MODEL_PATH_CSTR("RM_HP_59930007DR0130HP000.fbx"));
    //
    // {
    //     // CISDI_3DModelDataConverter converter {
    //     //     MODEL_PATH_CSTR("RM_HP_59930007DR0130HP000.fbx")};
    //     //
    //     // converter.Execute();
    //
    //     auto cisdiModelPath =
    //         MODEL_PATH_CSTR ("RM_HP_59930007DR0130HP000.cisdi");
    //
    //     auto meshes =
    //         CISDI_3DModelDataConverter::LoadCISDIModelData(cisdiModelPath);
    //
    //     mFactoryModel = MakeShared<Model>(meshes);
    //
    //     mFactoryModel->GenerateBuffers(mPContext.get(), this);
    // }

    {
        // mFactoryModel = MakeShared<Model>(MODEL_PATH_CSTR("sponza/sponza.obj"));
        // mFactoryModel = MakeShared<Model>(MODEL_PATH_CSTR("teapot.FBX"));
        // mFactoryModel = MakeShared<Model>(MODEL_PATH_CSTR("dragon.obj"), false);
        mFactoryModel =
            MakeShared<Model>(MODEL_PATH_CSTR("58360014DR2512ME021-2.STL"));

        mFactoryModel->GenerateMeshletBuffers(mPContext.get(), this);
    }

    CreateDescriptors();
    CreatePipelines();

#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif
}

EngineCore::~EngineCore() {
    mPContext->GetDeviceHandle().waitIdle();
}

void EngineCore::Run() {
    bool bQuit = false;

    while (!bQuit) {
        mPWindow->PollEvents(bQuit, mStopRendering, [&](SDL_Event* e) {
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
        mPSwapchain->GetImageHandle(mPSwapchain->AcquireNextImageIndex());

    const uint64_t graphicsFinished =
        mPContext->GetTimelineSemphore()->GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    // Compute Draw
    {
        auto cmd = mPCmdManager->GetCmdBufferToBegin();

        Utils::TransitionImageLayout(
            cmd.GetHandle(),
            mRenderResManager->GetResource("DrawImage")->GetTexHandle(),
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        DrawBackground(cmd.GetHandle());

        cmd.End();

        ::std::vector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             mPSwapchain->GetReady4RenderSemHandle(), 0ui64},
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             mPContext->GetTimelineSemaphoreHandle(), graphicsFinished}};

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mPContext->GetTimelineSemaphoreHandle(), computeFinished}};

        mPCmdManager->Submit(cmd.GetHandle(),
                             mPContext->GetDevice()->GetGraphicQueue(), waits,
                             signals);
    }

    // Graphics Draw
    {
        auto cmd = mPCmdManager->GetCmdBufferToBegin();

        Utils::TransitionImageLayout(
            cmd.GetHandle(),
            mRenderResManager->GetResource("DrawImage")->GetTexHandle(),
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eColorAttachmentOptimal);

        Utils::TransitionImageLayout(
            cmd.GetHandle(),
            mRenderResManager->GetResource("DepthImage")->GetTexHandle(),
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal);

        // DrawMesh(cmd.GetHandle());
        MeshShaderDraw(cmd.GetHandle());

        Utils::TransitionImageLayout(
            cmd.GetHandle(),
            mRenderResManager->GetResource("DrawImage")->GetTexHandle(),
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
             mPContext->GetTimelineSemaphoreHandle(), computeFinished}};

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mPContext->GetTimelineSemaphoreHandle(), allFinished},
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mPSwapchain->GetReady4PresentSemHandle()}};

        mPCmdManager->Submit(cmd.GetHandle(),
                             mPContext->GetDevice()->GetGraphicQueue(), waits,
                             signals);
    }

    {
        auto cmd = mPCmdManager->GetCmdBufferToBegin();
        cmd.End();

        ::std::vector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mPContext->GetTimelineSemaphoreHandle(), allFinished + 1}};

        mPCmdManager->Submit(cmd.GetHandle(),
                             mPContext->GetDevice()->GetGraphicQueue(), {},
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

    mPSwapchain->Present(mPContext->GetDevice()->GetGraphicQueue());

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
        mPWindow->GetVulkanInstanceExtension();
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
    enabledDeivceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

#ifdef CUDA_VULKAN_INTEROP
    enabledDeivceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#endif

    Context::EnableDefaultFeatures();

    return MakeUnique<Context>(
        mPWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

UniquePtr<Swapchain> EngineCore::CreateSwapchain() {
    return MakeUnique<Swapchain>(
        mPContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mPWindow->GetWidth()),
                      static_cast<uint32_t>(mPWindow->GetHeight())});
}

UniquePtr<RenderResourceManager> EngineCore::CreateRenderResourceManager() {
    return MakeUnique<RenderResourceManager>(mPContext->GetDevice(),
                                             mPContext->GetVmaAllocator());
}

void EngineCore::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mPWindow->GetWidth()),
                                  static_cast<uint32_t>(mPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto ptr = mRenderResManager->CreateTexture(
        "DrawImage", RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, drawImageExtent, drawImageUsage);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
}

void EngineCore::CreateDepthImage() {
    vk::Extent3D depthImageExtent {static_cast<uint32_t>(mPWindow->GetWidth()),
                                   static_cast<uint32_t>(mPWindow->GetHeight()),
                                   1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto ptr = mRenderResManager->CreateTexture(
        "DepthImage", RenderResource::Type::Texture2D, vk::Format::eD32Sfloat,
        depthImageExtent, depthImageUsage);
    ptr->CreateTexView("Depth-Whole", vk::ImageAspectFlagBits::eDepth);
}

UniquePtr<ImmediateSubmitManager> EngineCore::CreateImmediateSubmitManager() {
    return MakeUnique<ImmediateSubmitManager>(
        mPContext.get(),
        mPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

UniquePtr<CommandManager> EngineCore::CreateCommandManager() {
    return MakeUnique<CommandManager>(
        mPContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

void EngineCore::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
}

void EngineCore::CreateDescriptors() {

    CreateBackgroundComputeDescriptors();
    CreateMeshDescriptors();
    CreateDrawQuadDescriptors();
    CreateMeshShaderDescriptors();
}

void EngineCore::CreateErrorCheckTexture() {
    auto extent = VkExtent3D {16, 16, 1};
    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    auto ptr = mRenderResManager->CreateTexture(
        "ErrorCheckImage", RenderResource::Type::Texture2D,
        vk::Format::eR8G8B8A8Unorm, extent,
        vk::ImageUsageFlagBits::eSampled
            | vk::ImageUsageFlagBits::eTransferDst);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

    {
        size_t dataSize = extent.width * extent.height * 4;

        auto uploadBuffer = mPContext->CreateStagingBuffer(dataSize);
        memcpy(uploadBuffer->GetBufferMappedPtr(), pixels.data(), dataSize);

        mPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
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
}

UniquePtr<DescriptorManager> EngineCore::CreateDescriptorManager() {
    std::vector<DescPoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};

    return MakeUnique<DescriptorManager>(mPContext.get(), 10, sizes);
}

UniquePtr<PipelineManager> EngineCore::CreatePipelineManager() {
    return MakeUnique<PipelineManager>(mPContext.get());
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> EngineCore::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mPWindow->GetWidth()),
                                  static_cast<uint32_t>(mPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return mPContext->CreateExternalImage2D(
        drawImageExtent, vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::CreateExternalTriangleData() {
    mTriangleExternalMesh.mVertexBuffer =
        mPContext->CreateExternalPersistentBuffer(
            3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mTriangleExternalMesh.mIndexBuffer =
        mPContext->CreateExternalPersistentBuffer(
            3 * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);
}

void EngineCore::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore =
        MakeShared<CUDA::VulkanExternalSemaphore>(mPContext->GetDeviceHandle());
    mCUDASignalSemaphore =
        MakeShared<CUDA::VulkanExternalSemaphore>(mPContext->GetDeviceHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void EngineCore::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mPContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

void EngineCore::UpdateScene() {
    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(mPWindow->GetWidth())
                             / static_cast<float>(mPWindow->GetHeight()),
                         10000.0f, 0.0001f);

    proj[1][1] *= -1;

    mSceneData.cameraPos = glm::vec4 {mMainCamera.mPosition, 1.0f};
    mSceneData.view = view;
    mSceneData.proj = proj;
    mSceneData.viewProj = proj * view;
    UpdateSceneUBO();
}

void EngineCore::UpdateSceneUBO() {
    auto data = mRenderResManager->GetResource("SceneUniformBuffer")
                    ->GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
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

    mDescriptorManager->WriteImage(0,
                                   {VK_NULL_HANDLE,
                                    mRenderResManager->GetResource("DrawImage")
                                        ->GetTexViewHandle("Color-Whole"),
                                    vk::ImageLayout::eGeneral},
                                   vk::DescriptorType::eStorageImage);

    mDescriptorManager->WriteBuffer(
        1,
        {mRenderResManager->GetResource("RWBuffer")->GetBufferHandle(), 0,
         sizeof(glm::vec4) * mPWindow->GetWidth() * mPWindow->GetHeight()},
        vk::DescriptorType::eStorageBuffer);

    mDescriptorManager->UpdateSet(drawImageDesc);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void EngineCore::CreateBackgroundComputePipeline() {
    ::std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("DrawImage_Layout_0")};

    auto backgroundPipelineLayout = mPipelineManager->CreateLayout(
        "BackgoundCompute_Layout", setLayouts, {});

    Shader computeDrawShader {mPContext.get(), "computeDraw",
                              SHADER_PATH_CSTR("BackGround.comp.spv"),
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

    shaders.emplace_back(mPContext.get(), "vertex",
                         SHADER_PATH_CSTR("Triangle.vert.spv"),
                         ShaderStage::Vertex);

    shaders.emplace_back(mPContext.get(), "fragment",
                         SHADER_PATH_CSTR("Triangle.frag.spv"),
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
        .SetColorAttachmentFormat(
            mRenderResManager->GetResource("DrawImage")->GetTexFormat())
        .SetDepthStencilFormat(
            mRenderResManager->GetResource("DepthImage")->GetTexFormat())
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
    //     {mPContext->GetDefaultNearestSamplerHandle(),
    //      mErrorCheckImage->GetTexViewHandle("Color-Whole"),
    //      vk::ImageLayout::eShaderReadOnlyOptimal},
    //     vk::DescriptorType::eCombinedImageSampler);
    //
    // mDescriptorManager->UpdateSet(triangleDesc1);
}

void EngineCore::CreateMeshShaderPipeline() {
    std::vector<Shader> shaders;
    shaders.reserve(3);
    Type_ShaderMacros macros {};
    shaders.emplace_back(mPContext.get(), "Mesh shader fragment",
                         SHADER_PATH_CSTR("MeshShader.frag"),
                         ShaderStage::Fragment, false, macros);

    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    shaders.emplace_back(mPContext.get(), "Mesh shader task",
                         SHADER_PATH_CSTR("MeshShader.task"), ShaderStage::Task,
                         false, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    shaders.emplace_back(mPContext.get(), "Mesh shader mesh",
                         SHADER_PATH_CSTR("MeshShader.mesh"), ShaderStage::Mesh,
                         true, macros);

    vk::PushConstantRange meshPushConstants {};
    meshPushConstants.setSize(sizeof(PushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eMeshEXT);

    std::array setLayouts {
        mDescriptorManager->GetDescSetLayout("MeshShader_Desc_Layout_0")};

    auto meshShaderPipelineLayout = mPipelineManager->CreateLayout(
        "MeshShader_Pipe_Layout", setLayouts, {meshPushConstants});

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(meshShaderPipelineLayout->GetHandle())
        .SetShaders(shaders)
        // .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eLine)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(
            mRenderResManager->GetResource("DrawImage")->GetTexFormat())
        .SetDepthStencilFormat(
            mRenderResManager->GetResource("DepthImage")->GetTexFormat())
        .Build("MeshShaderDraw_Pipeline");

    DBG_LOG_INFO("Vulkan MeshShader Graphics Pipeline Created");
}

void EngineCore::CreateMeshShaderDescriptors() {
    // set = 0, binding = 0, scene data UBO
    {
        mDescriptorManager->AddDescSetLayoutBinding(
            0, 1, vk::DescriptorType::eUniformBuffer);

        const auto meshShaderSetLayout = mDescriptorManager->BuildDescSetLayout(
            "MeshShader_Desc_Layout_0", vk::ShaderStageFlagBits::eMeshEXT);

        const auto meshShaderDesc = mDescriptorManager->Allocate(
            "MeshShader_Desc_0_0", meshShaderSetLayout);

        mDescriptorManager->WriteBuffer(
            0,
            {mRenderResManager->GetResource("SceneUniformBuffer")
                 ->GetBufferHandle(),
             0, sizeof(SceneData)},
            vk::DescriptorType::eUniformBuffer);

        mDescriptorManager->UpdateSet(meshShaderDesc);
    }
}

void EngineCore::CreateDrawQuadDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto quadSetLayout = mDescriptorManager->BuildDescSetLayout(
        "Quad_Layout_0", vk::ShaderStageFlagBits::eFragment);

    const auto quadDesc =
        mDescriptorManager->Allocate("Quad_Desc_0", quadSetLayout);

    mDescriptorManager->WriteImage(0,
                                   {mPContext->GetDefaultLinearSamplerHandle(),
                                    mRenderResManager->GetResource("DrawImage")
                                        ->GetTexViewHandle("Color-Whole"),
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
                                   vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(quadDesc);
}

void EngineCore::CreateDrawQuadPipeline() {
    std::vector<Shader> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mPContext.get(), "vertex",
                         SHADER_PATH_CSTR("Quad.vert.spv"),
                         ShaderStage::Vertex);

    shaders.emplace_back(mPContext.get(), "fragment",
                         SHADER_PATH_CSTR("Quad.frag.spv"),
                         ShaderStage::Fragment);

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
        .SetColorAttachmentFormat(mPSwapchain->GetFormat())
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

    cmd.clearColorImage(
        mRenderResManager->GetResource("DrawImage")->GetTexHandle(),
        vk::ImageLayout::eGeneral, clearValue, subresource);

    // Compute Draw
    {
        cmd.bindPipeline(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetComputePipeline("BackgroundCompute_Pipeline"));

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetLayoutHandle("BackgoundCompute_Layout"), 0,
            mDescriptorManager->GetDescriptor("DrawImage_Desc_0"), {});

        cmd.dispatch(
            ::std::ceil(
                mRenderResManager->GetResource("DrawImage")->GetTexWidth()
                / 16.0),
            ::std::ceil(
                mRenderResManager->GetResource("DrawImage")->GetTexHeight()
                / 16.0),
            1);
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
    colorAttachment
        .setImageView(mRenderResManager->GetResource("DrawImage")
                          ->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment
        .setImageView(mRenderResManager->GetResource("DepthImage")
                          ->GetTexViewHandle("Depth-Whole"))
        .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {
            {0, 0},
            {mRenderResManager->GetResource("DrawImage")->GetTexWidth(),
             mRenderResManager->GetResource("DrawImage")->GetTexHeight()}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("TriangleDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport
        .setWidth(mRenderResManager->GetResource("DrawImage")->GetTexWidth())
        .setHeight(mRenderResManager->GetResource("DrawImage")->GetTexHeight())
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mRenderResManager->GetResource("DrawImage")->GetTexWidth(),
         mRenderResManager->GetResource("DrawImage")->GetTexHeight()});
    cmd.setScissor(0, scissor);

    mDescriptorManager->WriteImage(
        0,
        {mPContext->GetDefaultNearestSamplerHandle(),
         mRenderResManager->GetResource("ErrorCheckImage")
             ->GetTexViewHandle("Color-Whole"),
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

    cmd.bindIndexBuffer(
        mFactoryModel->GetMeshBuffer().mIndexBuffer->GetBufferHandle(), 0,
        vk::IndexType::eUint32);

    {
        glm::mat4 model {1.0f};
        // model = glm::scale(model, glm::vec3 {0.0001f});

        auto pushContants = mFactoryModel->GetPushContants();
        pushContants.mModelMatrix = model;

        cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
                          vk::ShaderStageFlagBits::eVertex, 0,
                          sizeof(pushContants), &pushContants);

        mDescriptorManager->WriteBuffer(
            0,
            {mRenderResManager->GetResource("SceneUniformBuffer")
                 ->GetBufferHandle(),
             0, sizeof(SceneData)},
            vk::DescriptorType::eUniformBuffer);

        mDescriptorManager->UpdateSet(
            mDescriptorManager->GetDescriptor("Triangle_Desc_0"));

        cmd.drawIndexedIndirect(
            mFactoryModel->GetIndexedIndirectCmdBuffer()->GetBufferHandle(), 0,
            mFactoryModel->GetMeshes().size(),
            sizeof(vk::DrawIndexedIndirectCommand));
    }

    cmd.endRendering();
}

void EngineCore::DrawQuad(vk::CommandBuffer cmd) {
    auto imageIndex = mPSwapchain->GetCurrentImageIndex();
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mPSwapchain->GetImageViewHandle(imageIndex))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {{0, 0},
                                   {mPSwapchain->GetExtent2D().width,
                                    mPSwapchain->GetExtent2D().height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("QuadDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport.setWidth(mPSwapchain->GetExtent2D().width)
        .setHeight(mPSwapchain->GetExtent2D().height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mPSwapchain->GetExtent2D().width, mPSwapchain->GetExtent2D().height});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           mPipelineManager->GetLayoutHandle("Quad_Layout"), 0,
                           mDescriptorManager->GetDescriptor("Quad_Desc_0"),
                           {});

    cmd.draw(3, 1, 0, 0);

    cmd.endRendering();
}

void EngineCore::MeshShaderDraw(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment
        .setImageView(mRenderResManager->GetResource("DrawImage")
                          ->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment
        .setImageView(mRenderResManager->GetResource("DepthImage")
                          ->GetTexViewHandle("Depth-Whole"))
        .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {
            {0, 0},
            {mRenderResManager->GetResource("DrawImage")->GetTexWidth(),
             mRenderResManager->GetResource("DrawImage")->GetTexHeight()}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("MeshShaderDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport
        .setWidth(mRenderResManager->GetResource("DrawImage")->GetTexWidth())
        .setHeight(mRenderResManager->GetResource("DrawImage")->GetTexHeight())
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mRenderResManager->GetResource("DrawImage")->GetTexWidth(),
         mRenderResManager->GetResource("DrawImage")->GetTexHeight()});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("MeshShader_Pipe_Layout"), 0,
        {mDescriptorManager->GetDescriptor("MeshShader_Desc_0_0")}, {});

    auto meshPushContants = mFactoryModel->GetPushContants();
    meshPushContants.mModelMatrix =
        glm::scale(meshPushContants.mModelMatrix, glm::vec3 {0.0001f});
    cmd.pushConstants(
        mPipelineManager->GetLayoutHandle("MeshShader_Pipe_Layout"),
        vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(meshPushContants),
        &meshPushContants);

    // uint32_t taskCount =
    //     (mFactoryModel->GetMeshletCount() + TASK_SHADER_INVOCATION_COUNT - 1)
    //     / TASK_SHADER_INVOCATION_COUNT;
    // cmd.drawMeshTasksEXT(taskCount, 1, 1);
    
    cmd.drawMeshTasksIndirectEXT(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetBufferHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));

    cmd.endRendering();
}

}  // namespace IntelliDesign_NS::Vulkan::Core