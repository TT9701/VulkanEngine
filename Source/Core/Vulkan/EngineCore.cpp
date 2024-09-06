#include "EngineCore.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Platform/Window.hpp"
#include "Core/Vulkan/Manager/Context.hpp"
#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Core/Vulkan/Manager/PipelineManager.hpp"
#include "Core/Vulkan/Manager/RenderResourceManager.hpp"
#include "Core/Vulkan/Manager/ShaderManager.hpp"
#include "Core/Vulkan/Native/Buffer.hpp"
#include "Core/Vulkan/Native/RenderResource.hpp"
#include "Core/Vulkan/Native/Shader.hpp"
#include "Core/Vulkan/Native/Swapchain.hpp"

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
      mDescriptorManager(CreateDescriptorBufferManager()),
      mPipelineManager(CreatePipelineManager()),
      mShaderModuleManager(CreateShaderModuleManager()) {
    CreateDrawImage();
    CreateDepthImage();
    CreateErrorCheckTexture();

    mRenderResManager->CreateBuffer(
        "SceneUniformBuffer", sizeof(SceneData),
        vk::BufferUsageFlagBits::eUniformBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::Staging);

    mRenderResManager->CreateBuffer(
        "RWBuffer",
        sizeof(glm::vec4) * mPWindow->GetWidth() * mPWindow->GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::DeviceLocal);

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // models: teapot.FBX sphere.fbx dragon.obj buddha.obj sponza/sponza.obj
    //         RM_HP_59930007DR0130HP000.fbx Foliage.fbx
    {
        // mFactoryModel = MakeShared<Model>(MODEL_PATH_CSTR("sponza/sponza.obj"));

        // CISDI_3DModelDataConverter converter {
        //     MODEL_PATH_CSTR("sponza/sponza.obj")};
        //
        // converter.Execute();

        auto cisdiModelPath = MODEL_PATH("RM_HP_59930007DR0130HP000.cisdi");

        auto meshes = CISDI_3DModelDataConverter::LoadCISDIModelData(
            cisdiModelPath.c_str());

        mFactoryModel = MakeShared<Model>(meshes);

        // mFactoryModel->GenerateBuffers(mPContext.get(), this);
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

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             mPSwapchain->GetReady4RenderSemHandle(), 0ui64},
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             mPContext->GetTimelineSemaphoreHandle(), graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
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

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eComputeShader,
             mPContext->GetTimelineSemaphoreHandle(), computeFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
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

        Type_STLVector<SemSubmitInfo> signals = {
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
    Type_STLVector<Type_STLString> requestedInstanceLayers {};
#ifndef NDEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    auto sdlRequestedInstanceExtensions =
        mPWindow->GetVulkanInstanceExtension();
    Type_STLVector<Type_STLString> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifndef NDEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    Type_STLVector<Type_STLString> enabledDeivceExtensions {};

    enabledDeivceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_MAINTENANCE_6_EXTENSION_NAME);

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

        auto uploadBuffer = mPContext->CreateStagingBuffer("", dataSize);
        memcpy(uploadBuffer->GetMapPtr(), pixels.data(), dataSize);

        mPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
            Utils::TransitionImageLayout(cmd, ptr->GetTexHandle(),
                                         vk::ImageLayout::eUndefined,
                                         vk::ImageLayout::eTransferDstOptimal);

            vk::BufferImageCopy copyRegion {};
            copyRegion
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(extent);

            cmd.copyBufferToImage(
                uploadBuffer->GetHandle(), ptr->GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal, copyRegion);

            Utils::TransitionImageLayout(
                cmd, ptr->GetTexHandle(), vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        });
    }
}

UniquePtr<PipelineManager> EngineCore::CreatePipelineManager() {
    return MakeUnique<PipelineManager>(mPContext.get());
}

UniquePtr<ShaderManager> EngineCore::CreateShaderModuleManager() {
    return MakeUnique<ShaderManager>(mPContext.get());
}

UniquePtr<DescriptorManager> EngineCore::CreateDescriptorBufferManager() {
    return MakeUnique<DescriptorManager>(mPContext.get());
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
    Type_STLVector<vk::DescriptorSetLayoutBinding> bindings {};
    bindings.emplace_back(0, vk::DescriptorType::eStorageImage, 1,
                          vk::ShaderStageFlagBits::eCompute);
    bindings.emplace_back(1, vk::DescriptorType::eStorageBuffer, 1,
                          vk::ShaderStageFlagBits::eCompute);

    auto setLayout =
        mDescriptorManager->CreateDescLayout("DrawImage_Desc_Layout", bindings);

    mDescriptorManager->CreateDescriptorSet("DrawImage",
                                            "DrawImage_Desc_Layout", 0);

    vk::DescriptorImageInfo imageInfo {};
    imageInfo
        .setImageView(mRenderResManager->GetResource("DrawImage")
                          ->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eGeneral);

    mDescriptorManager->CreateImageDescriptor(
        mDescriptorManager->GetDescriptorSet("DrawImage"), 0,
        vk::DescriptorType::eStorageImage, &imageInfo);

    vk::DescriptorAddressInfoEXT bufferInfo {};
    bufferInfo
        .setAddress(mRenderResManager->GetResource("RWBuffer")
                        ->GetBufferDeviceAddress())
        .setRange(mRenderResManager->GetResource("RWBuffer")->GetBufferSize());

    mDescriptorManager->CreateBufferDescriptor(
        mDescriptorManager->GetDescriptorSet("DrawImage"), 1,
        vk::DescriptorType::eStorageBuffer, &bufferInfo);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void EngineCore::CreateBackgroundComputePipeline() {
    ::std::array setLayouts {
        mDescriptorManager->GetDescSetLayoutHandle("DrawImage_Desc_Layout")};

    auto backgroundPipelineLayout = mPipelineManager->CreateLayout(
        "BackgoundCompute_Layout", setLayouts, {});

    auto computeDrawShader = mShaderModuleManager->CreateShaderFromSource(
        "computeDraw", SHADER_PATH_CSTR("BackGround.comp"),
        ShaderStage::Compute);

    auto& builder = mPipelineManager->GetComputePipelineBuilder();

    auto backgroundComputePipeline =
        builder
            .SetShader(mShaderModuleManager->GetShader("computeDraw",
                                                       ShaderStage::Compute))
            .SetLayout(backgroundPipelineLayout->GetHandle())
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("BackgroundCompute_Pipeline");

    mShaderModuleManager->ReleaseShader("computeDraw", ShaderStage::Compute);

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void EngineCore::CreateMeshPipeline() {
    Type_STLVector<SharedPtr<Shader>> shaders;
    shaders.reserve(2);
    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "vertex", SHADER_PATH_CSTR("Triangle.vert"), ShaderStage::Vertex,
        true));
    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "fragment", SHADER_PATH_CSTR("Triangle.frag"), ShaderStage::Fragment,
        true));

    Type_STLVector<vk::PushConstantRange> pushConstants(1);
    pushConstants[0]
        .setSize(sizeof(PushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    ::std::array setLayouts {
        mDescriptorManager->GetDescSetLayoutHandle("DrawMesh_Desc_Layout_0"),
        mDescriptorManager->GetDescSetLayoutHandle("DrawMesh_Desc_Layout_1")};

    auto trianglePipelineLayout = mPipelineManager->CreateLayout(
        "Triangle_Layout", setLayouts, pushConstants);

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
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("TriangleDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void EngineCore::CreateMeshDescriptors() {
    // Type_STLVector<vk::DescriptorSetLayoutBinding> bindings {};
    // bindings.emplace_back(
    //     0, vk::DescriptorType::eUniformBuffer, 1,
    //     vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    // auto setLayout0 =
    //     mDescriptorManager->CreateDescLayout("DrawMesh_0", bindings);
    //
    // bindings.clear();
    // bindings.emplace_back(0, vk::DescriptorType::eCombinedImageSampler, 1,
    //                       vk::ShaderStageFlagBits::eFragment);
    // auto setLayout1 =
    //     mDescriptorManager->CreateDescLayout("DrawMesh_1", bindings);

    mDescriptorManager->CreateDescLayouts(
        "DrawMesh_Desc_Layout",
        {{0, 0, vk::DescriptorType::eUniformBuffer,
          vk::ShaderStageFlagBits::eVertex
              | vk::ShaderStageFlagBits::eFragment},
         {1, 0, vk::DescriptorType::eCombinedImageSampler,
          vk::ShaderStageFlagBits::eFragment}});

    auto set0 = mDescriptorManager->CreateDescriptorSet(
        "DrawMesh_DescSet_0", "DrawMesh_Desc_Layout_0", 0);

    auto set1 = mDescriptorManager->CreateDescriptorSet(
        "DrawMesh_DescSet_1", "DrawMesh_Desc_Layout_1", 0);

    vk::DescriptorAddressInfoEXT bufferInfo {};
    bufferInfo
        .setAddress(mRenderResManager->GetResource("SceneUniformBuffer")
                        ->GetBufferDeviceAddress())
        .setRange(mRenderResManager->GetResource("SceneUniformBuffer")
                      ->GetBufferSize());

    mDescriptorManager->CreateBufferDescriptor(
        set0.get(), 0, vk::DescriptorType::eUniformBuffer, &bufferInfo);

    vk::DescriptorImageInfo imageInfo {};
    imageInfo
        .setImageView(mRenderResManager->GetResource("ErrorCheckImage")
                          ->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSampler(mPContext->GetDefaultNearestSamplerHandle());

    mDescriptorManager->CreateImageDescriptor(
        set1.get(), 0, vk::DescriptorType::eCombinedImageSampler, &imageInfo);
}

void EngineCore::CreateMeshShaderPipeline() {
    Type_STLVector<SharedPtr<Shader>> shaders;
    shaders.reserve(3);
    Type_ShaderMacros macros {};
    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "Mesh shader fragment", SHADER_PATH_CSTR("MeshShader.frag"),
        ShaderStage::Fragment));

    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        ShaderStage::Task, false, macros));

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        ShaderStage::Mesh, true, macros));

    Type_STLVector<vk::PushConstantRange> meshPushConstants(1);
    meshPushConstants[0]
        .setSize(sizeof(PushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eMeshEXT);

    std::array setLayouts {mDescriptorManager->GetDescSetLayoutHandle(
        "MeshShaderDraw_Desc_Layout")};

    auto meshShaderPipelineLayout = mPipelineManager->CreateLayout(
        "MeshShader_Pipe_Layout", setLayouts, meshPushConstants);

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(meshShaderPipelineLayout->GetHandle())
        .SetShaders(shaders)
        // .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(
            mRenderResManager->GetResource("DrawImage")->GetTexFormat())
        .SetDepthStencilFormat(
            mRenderResManager->GetResource("DepthImage")->GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("MeshShaderDraw_Pipeline");

    DBG_LOG_INFO("Vulkan MeshShader Graphics Pipeline Created");
}

void EngineCore::CreateMeshShaderDescriptors() {
    // set = 0, binding = 0, scene data UBO
    Type_STLVector<vk::DescriptorSetLayoutBinding> bindings {};
    bindings.emplace_back(0, vk::DescriptorType::eUniformBuffer, 1,
                          vk::ShaderStageFlagBits::eMeshEXT);
    auto setLayout = mDescriptorManager->CreateDescLayout(
        "MeshShaderDraw_Desc_Layout", bindings);

    auto set = mDescriptorManager->CreateDescriptorSet(
        "MeshShaderDraw", "MeshShaderDraw_Desc_Layout", 0);

    vk::DescriptorAddressInfoEXT bufferInfo {};
    bufferInfo
        .setAddress(mRenderResManager->GetResource("SceneUniformBuffer")
                        ->GetBufferDeviceAddress())
        .setRange(mRenderResManager->GetResource("SceneUniformBuffer")
                      ->GetBufferSize());

    mDescriptorManager->CreateBufferDescriptor(
        set.get(), 0, vk::DescriptorType::eUniformBuffer, &bufferInfo);
}

void EngineCore::CreateDrawQuadDescriptors() {
    Type_STLVector<vk::DescriptorSetLayoutBinding> bindings {};
    bindings.emplace_back(0, vk::DescriptorType::eCombinedImageSampler, 1,
                          vk::ShaderStageFlagBits::eFragment);
    auto setLayout =
        mDescriptorManager->CreateDescLayout("DrawQuad_Desc_Layout", bindings);

    auto set = mDescriptorManager->CreateDescriptorSet(
        "DrawQuad", "DrawQuad_Desc_Layout", 0);

    vk::DescriptorImageInfo imageInfo {};
    imageInfo
        .setImageView(mRenderResManager->GetResource("DrawImage")
                          ->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSampler(mPContext->GetDefaultLinearSamplerHandle());

    mDescriptorManager->CreateImageDescriptor(
        set.get(), 0, vk::DescriptorType::eCombinedImageSampler, &imageInfo);
}

void EngineCore::CreateDrawQuadPipeline() {
    Type_STLVector<SharedPtr<Shader>> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "Quad vertex", SHADER_PATH_CSTR("Quad.vert"), ShaderStage::Vertex));

    shaders.emplace_back(mShaderModuleManager->CreateShaderFromSource(
        "Quad fragment", SHADER_PATH_CSTR("Quad.frag"), ShaderStage::Fragment));

    ::std::array setLayouts {
        mDescriptorManager->GetDescSetLayoutHandle("DrawQuad_Desc_Layout")};

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
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
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
        mPipelineManager->BindComputePipeline(cmd,
                                              "BackgroundCompute_Pipeline");

        ::std::array bufferIndex {0ui32};
        mDescriptorManager->BindDescBuffers(cmd, bufferIndex);

        ::std::array setBufferIndex = {0ui32};
        ::std::array<Type_STLString, 1> descSetNames {"DrawImage"};
        mDescriptorManager->BindDescriptorSets(
            cmd, vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetLayoutHandle("BackgoundCompute_Layout"), 0ui32,
            setBufferIndex, descSetNames);

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

    mPipelineManager->BindGraphicsPipeline(cmd, "TriangleDraw_Pipeline");

    ::std::array bufferIndex {0ui32};
    mDescriptorManager->BindDescBuffers(cmd, bufferIndex);

    ::std::array setBufferIndices = {0ui32, 0ui32};
    ::std::array<Type_STLString, 2> descSetNames {"DrawMesh_DescSet_0",
                                                  "DrawMesh_DescSet_1"};
    mDescriptorManager->BindDescriptorSets(
        cmd, vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("Triangle_Layout"), 0ui32,
        setBufferIndices, descSetNames);

    cmd.bindIndexBuffer(
        mFactoryModel->GetMeshBuffer().mIndexBuffer->GetBufferHandle(), 0,
        vk::IndexType::eUint32);

    {
        glm::mat4 model {1.0f};
        model = glm::scale(model, glm::vec3 {0.0001f});
        auto pushContants = mFactoryModel->GetPushContants();
        pushContants.mModelMatrix = model;

        cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
                          vk::ShaderStageFlagBits::eVertex, 0,
                          sizeof(pushContants), &pushContants);

        cmd.drawIndexedIndirect(
            mFactoryModel->GetIndexedIndirectCmdBuffer()->GetHandle(), 0,
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

    mPipelineManager->BindGraphicsPipeline(cmd, "QuadDraw_Pipeline");

    ::std::array bufferIndex {0ui32};
    mDescriptorManager->BindDescBuffers(cmd, bufferIndex);

    ::std::array setBufferIndices = {0ui32};
    ::std::array<Type_STLString, 1> descSetNames {"DrawQuad"};
    mDescriptorManager->BindDescriptorSets(
        cmd, vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("Quad_Layout"), 0ui32,
        setBufferIndices, descSetNames);

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

    mPipelineManager->BindGraphicsPipeline(cmd, "MeshShaderDraw_Pipeline");

    ::std::array bufferIndex {0ui32};
    mDescriptorManager->BindDescBuffers(cmd, bufferIndex);

    ::std::array setBufferIndices = {0ui32};
    ::std::array<Type_STLString, 1> descSetNames {"MeshShaderDraw"};
    mDescriptorManager->BindDescriptorSets(
        cmd, vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("MeshShader_Pipe_Layout"), 0ui32,
        setBufferIndices, descSetNames);

    auto meshPushContants = mFactoryModel->GetPushContants();
    meshPushContants.mModelMatrix =
        glm::scale(meshPushContants.mModelMatrix, glm::vec3 {0.0001f});
    cmd.pushConstants(
        mPipelineManager->GetLayoutHandle("MeshShader_Pipe_Layout"),
        vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(meshPushContants),
        &meshPushContants);

    cmd.drawMeshTasksIndirectEXT(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));

    cmd.endRendering();
}

}  // namespace IntelliDesign_NS::Vulkan::Core