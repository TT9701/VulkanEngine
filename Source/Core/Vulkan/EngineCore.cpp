#include "EngineCore.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Platform/Window.hpp"
#include "Core/Vulkan/Manager/Context.hpp"
#include "Core/Vulkan/Native/Swapchain.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

EngineCore::EngineCore()
    : mWindow(CreateSDLWindow()),
      mContext(CreateContext()),
      mSwapchain(CreateSwapchain()),
      mDescMgr(CreateDescriptorManager()),
      mRenderResMgr(CreateRenderResourceManager()),
      mCmdMgr(CreateCommandManager()),
      mImmSubmitMgr(CreateImmediateSubmitManager()),
      mPipelineMgr(CreatePipelineManager()),
      mShaderMgr(CreateShaderManager()),
#ifdef CUDA_VULKAN_INTEROP
      mCUDAExternalImage(CreateExternalImage())
#endif
{
    CreateDrawImage();
    CreateDepthImage();

    LoadShaders();
    CreatePipelines();

    CreateErrorCheckTexture();

    {
        mRenderResMgr.CreateBuffer(
            "SceneUniformBuffer", sizeof(SceneData),
            vk::BufferUsageFlagBits::eUniformBuffer
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            Buffer::MemoryType::Staging);

        mRenderResMgr.CreateDescriptorSet(
            "Triangle_Scene_Data", 0, "TriangleDraw",
            vk::ShaderStageFlagBits::eVertex
                | vk::ShaderStageFlagBits::eFragment,
            {{0, "SceneUniformBuffer", vk::DescriptorType::eUniformBuffer,
              "SceneDataUBO"}});

        mRenderResMgr.CreateDescriptorSet(
            "MeshShader_Scene_Data", 0, "MeshShaderDraw",
            vk::ShaderStageFlagBits::eMeshEXT,
            {{0, "SceneUniformBuffer", vk::DescriptorType::eUniformBuffer,
              "UBO"}});
    }

    {
        mRenderResMgr.CreateScreenSizeBuffer(
            "RWBuffer",
            sizeof(glm::vec4) * mWindow->GetWidth() * mWindow->GetHeight(),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            Buffer::MemoryType::DeviceLocal, sizeof(glm::vec4));

        mRenderResMgr.CreateDescriptorSet(
            "Storage_Image_Buffer", 0, "Background",
            vk::ShaderStageFlagBits::eCompute,
            {{0, "DrawImage", vk::DescriptorType::eStorageImage, "image",
              "Color-Whole"},
             {1, "RWBuffer", vk::DescriptorType::eStorageBuffer,
              "StorageBuffer"}});
    }

    {
        mRenderResMgr.CreateDescriptorSet(
            "DrawImage_Texture", 0, "QuadDraw",
            vk::ShaderStageFlagBits::eFragment,
            {{0, "DrawImage", vk::DescriptorType::eCombinedImageSampler, "tex0",
              "Color-Whole", mContext->GetDefaultLinearSampler()}});
    }

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

        // mFactoryModel->GenerateBuffers(mContext.get(), this);
        mFactoryModel->GenerateMeshletBuffers(mContext.get(), this);
    }

    RecordDrawBackgroundCmds();
    // RecordDrawMeshCmds();
    RecordMeshShaderDrawCmds();
    RecordDrawQuadCmds();

#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif
}

EngineCore::~EngineCore() {
    mContext->GetDeviceHandle().waitIdle();
}

void EngineCore::Run() {
    bool bQuit = false;

    while (!bQuit) {
        mWindow->PollEvents(
            bQuit, mStopRendering,
            [&](SDL_Event* e) { mMainCamera.ProcessSDLEvent(e, 0.001f); },
            [&]() {
                mSwapchain->bResizeRequested = true;
                mSwapchain->Resize(mWindow->GetWidth(), mWindow->GetHeight());

                mRenderResMgr.ResizeScreenSizeResources(mWindow->GetWidth(),
                                                        mWindow->GetHeight());
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
    auto swapchainImageIdx = mSwapchain->AcquireNextImageIndex();
    if (swapchainImageIdx == -1)
        return;

    auto swapchainImage = mSwapchain->GetImageHandle(swapchainImageIdx);

    const uint64_t graphicsFinished =
        mContext->GetTimelineSemphore()->GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    // Compute Draw
    {
        auto cmd = mCmdMgr.GetCmdBufferToBegin();

        mBackgroundDrawCallMgr.RecordCmd(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             mSwapchain->GetReady4RenderSemHandle(), 0ui64},
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             mContext->GetTimelineSemaphoreHandle(), graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mContext->GetTimelineSemaphoreHandle(), computeFinished}};

        mCmdMgr.Submit(cmd.GetHandle(),
                       mContext->GetDevice()->GetGraphicQueue(), waits,
                       signals);
    }

    // Graphics Draw
    {
        auto cmd = mCmdMgr.GetCmdBufferToBegin();

        // mMeshDrawCallMgr.RecordCmd(cmd.GetHandle());
        mMeshShaderDrawCallMgr.RecordCmd(cmd.GetHandle());

        mQuadDrawCallMgr.UpdateArgument_Attachments(
            {0}, {mSwapchain->GetColorAttachmentInfo(swapchainImageIdx)});
        mQuadDrawCallMgr.UpdateArgument_ImageBarriers_BeforePass(
            {"Swapchain"},
            {mSwapchain->GetImageBarrier_BeforePass(swapchainImageIdx)});
        mQuadDrawCallMgr.UpdateArgument_ImageBarriers_AfterPass(
            {"Swapchain"},
            {mSwapchain->GetImageBarrier_AfterPass(swapchainImageIdx)});

        mQuadDrawCallMgr.RecordCmd(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eComputeShader,
             mContext->GetTimelineSemaphoreHandle(), computeFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mContext->GetTimelineSemaphoreHandle(), allFinished},
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mSwapchain->GetReady4PresentSemHandle()}};

        mCmdMgr.Submit(cmd.GetHandle(),
                       mContext->GetDevice()->GetGraphicQueue(), waits,
                       signals);
    }

    {
        auto cmd = mCmdMgr.GetCmdBufferToBegin();
        cmd.End();

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mContext->GetTimelineSemaphoreHandle(), allFinished + 1}};

        mCmdMgr.Submit(cmd.GetHandle(),
                       mContext->GetDevice()->GetGraphicQueue(), {}, signals);
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

    mSwapchain->Present(mContext->GetDevice()->GetGraphicQueue());

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

    auto sdlRequestedInstanceExtensions = mWindow->GetVulkanInstanceExtension();
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
        mWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

UniquePtr<Swapchain> EngineCore::CreateSwapchain() {
    return MakeUnique<Swapchain>(
        mContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mWindow->GetWidth()),
                      static_cast<uint32_t>(mWindow->GetHeight())});
}

RenderResourceManager EngineCore::CreateRenderResourceManager() {
    return {mContext->GetDevice(), mContext->GetVmaAllocator(), &mDescMgr};
}

void EngineCore::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                  static_cast<uint32_t>(mWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto ptr = mRenderResMgr.CreateScreenSizeTexture(
        "DrawImage", RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, drawImageExtent, drawImageUsage);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
}

void EngineCore::CreateDepthImage() {
    vk::Extent3D depthImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                   static_cast<uint32_t>(mWindow->GetHeight()),
                                   1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto ptr = mRenderResMgr.CreateScreenSizeTexture(
        "DepthImage", RenderResource::Type::Texture2D,
        vk::Format::eD24UnormS8Uint, depthImageExtent, depthImageUsage);
    ptr->CreateTexView("Depth-Whole", vk::ImageAspectFlagBits::eDepth
                                          | vk::ImageAspectFlagBits::eStencil);

    mImmSubmitMgr.Submit([&](vk::CommandBuffer cmd) {
        Utils::TransitionImageLayout(
            cmd, ptr->GetTexHandle(), vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    });
}

ImmediateSubmitManager EngineCore::CreateImmediateSubmitManager() {
    return {
        mContext.get(),
        mContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value()};
}

CommandManager EngineCore::CreateCommandManager() {
    return {
        mContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value()};
}

void EngineCore::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
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

    auto ptr = mRenderResMgr.CreateTexture(
        "ErrorCheckImage", RenderResource::Type::Texture2D,
        vk::Format::eR8G8B8A8Unorm, extent,
        vk::ImageUsageFlagBits::eSampled
            | vk::ImageUsageFlagBits::eTransferDst);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

    {
        size_t dataSize = extent.width * extent.height * 4;

        auto uploadBuffer = mContext->CreateStagingBuffer("", dataSize);
        memcpy(uploadBuffer->GetMapPtr(), pixels.data(), dataSize);

        mImmSubmitMgr.Submit([&](vk::CommandBuffer cmd) {
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

    {
        mRenderResMgr.CreateDescriptorSet(
            "ErrorCheck_Image", 1, "TriangleDraw",
            vk::ShaderStageFlagBits::eFragment,
            {{0, "ErrorCheckImage", vk::DescriptorType::eCombinedImageSampler,
              "tex0", "Color-Whole", mContext->GetDefaultNearestSampler()}});
    }
}

PipelineManager EngineCore::CreatePipelineManager() {
    return {mContext.get()};
}

ShaderManager EngineCore::CreateShaderManager() {
    return {mContext.get()};
}

DescriptorManager EngineCore::CreateDescriptorManager() {
    return {mContext.get()};
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> EngineCore::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                  static_cast<uint32_t>(mWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return mContext->CreateExternalImage2D(
        drawImageExtent, vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::CreateExternalTriangleData() {
    mTriangleExternalMesh.mVertexBuffer =
        mContext->CreateExternalPersistentBuffer(
            3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mTriangleExternalMesh.mIndexBuffer =
        mContext->CreateExternalPersistentBuffer(
            3 * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);
}

void EngineCore::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore =
        MakeShared<CUDA::VulkanExternalSemaphore>(mContext->GetDeviceHandle());
    mCUDASignalSemaphore =
        MakeShared<CUDA::VulkanExternalSemaphore>(mContext->GetDeviceHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void EngineCore::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

void EngineCore::UpdateScene() {
    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(mWindow->GetWidth())
                             / static_cast<float>(mWindow->GetHeight()),
                         10000.0f, 0.0001f);

    proj[1][1] *= -1;

    mSceneData.cameraPos = glm::vec4 {mMainCamera.mPosition, 1.0f};
    mSceneData.view = view;
    mSceneData.proj = proj;
    mSceneData.viewProj = proj * view;
    UpdateSceneUBO();
}

void EngineCore::UpdateSceneUBO() {
    auto data = mRenderResMgr["SceneUniformBuffer"]->GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

void EngineCore::LoadShaders() {
    mShaderMgr.CreateShaderFromSource("computeDraw",
                                      SHADER_PATH_CSTR("BackGround.comp"),
                                      vk::ShaderStageFlagBits::eCompute);

    mShaderMgr.CreateShaderFromSource("vertex",
                                      SHADER_PATH_CSTR("Triangle.vert"),
                                      vk::ShaderStageFlagBits::eVertex, true);

    mShaderMgr.CreateShaderFromSource("fragment",
                                      SHADER_PATH_CSTR("Triangle.frag"),
                                      vk::ShaderStageFlagBits::eFragment, true);

    mShaderMgr.CreateShaderFromSource("Mesh shader fragment",
                                      SHADER_PATH_CSTR("MeshShader.frag"),
                                      vk::ShaderStageFlagBits::eFragment);

    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    mShaderMgr.CreateShaderFromSource(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        vk::ShaderStageFlagBits::eTaskEXT, false, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    mShaderMgr.CreateShaderFromSource(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        vk::ShaderStageFlagBits::eMeshEXT, true, macros);

    mShaderMgr.CreateShaderFromSource("Quad vertex",
                                      SHADER_PATH_CSTR("Quad.vert"),
                                      vk::ShaderStageFlagBits::eVertex);

    mShaderMgr.CreateShaderFromSource("Quad fragment",
                                      SHADER_PATH_CSTR("Quad.frag"),
                                      vk::ShaderStageFlagBits::eFragment);
}

void EngineCore::CreateBackgroundComputePipeline() {
    auto builder = mPipelineMgr.GetComputePipelineBuilder(&mDescMgr);

    auto backgroundComputePipeline =
        builder
            .SetShader(mShaderMgr.GetShader("computeDraw",
                                            vk::ShaderStageFlagBits::eCompute))
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void EngineCore::CreateMeshPipeline() {
    Type_STLVector<Shader*> shaders;
    shaders.reserve(2);
    shaders.emplace_back(
        mShaderMgr.GetShader("vertex", vk::ShaderStageFlagBits::eVertex));
    shaders.emplace_back(
        mShaderMgr.GetShader("fragment", vk::ShaderStageFlagBits::eFragment));

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder(&mDescMgr);
    builder.SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eFront,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(mRenderResMgr["DrawImage"]->GetTexFormat())
        .SetDepthStencilFormat(mRenderResMgr["DepthImage"]->GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("TriangleDraw");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void EngineCore::CreateMeshShaderPipeline() {
    Type_STLVector<Shader*> shaders;
    shaders.reserve(3);
    shaders.emplace_back(mShaderMgr.GetShader(
        "Mesh shader fragment", vk::ShaderStageFlagBits::eFragment));

    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    shaders.emplace_back(mShaderMgr.GetShader(
        "Mesh shader task", vk::ShaderStageFlagBits::eTaskEXT, macros));

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    shaders.emplace_back(mShaderMgr.GetShader(
        "Mesh shader mesh", vk::ShaderStageFlagBits::eMeshEXT, macros));

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder(&mDescMgr);
    builder.SetShaders(shaders)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(mRenderResMgr["DrawImage"]->GetTexFormat())
        .SetDepthStencilFormat(mRenderResMgr["DepthImage"]->GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("MeshShaderDraw");

    DBG_LOG_INFO("Vulkan MeshShader Graphics Pipeline Created");
}

void EngineCore::CreateDrawQuadPipeline() {
    Type_STLVector<Shader*> shaders;
    shaders.reserve(2);
    shaders.emplace_back(
        mShaderMgr.GetShader("Quad vertex", vk::ShaderStageFlagBits::eVertex));
    shaders.emplace_back(mShaderMgr.GetShader(
        "Quad fragment", vk::ShaderStageFlagBits::eFragment));

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder(&mDescMgr);
    builder.SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::False, vk::False)
        .SetColorAttachmentFormat(mSwapchain->GetFormat())
        .SetDepthStencilFormat(vk::Format::eUndefined)
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("QuadDraw");

    DBG_LOG_INFO("Vulkan Quad Graphics Pipeline Created");
}

void EngineCore::RecordDrawBackgroundCmds() {
    vk::ImageMemoryBarrier2 drawImageBarrier {
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderStorageWrite,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral,
        {},
        {},
        mRenderResMgr["DrawImage"]->GetTexHandle(),
        Utils::GetWholeImageSubresource(vk::ImageAspectFlagBits::eColor)};
    mBackgroundDrawCallMgr.AddArgument_MemoryBarriers_BeforePass(
        {"DrawImage"}, {drawImageBarrier});

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));

    vk::ClearColorValue clearValue {flash, flash, flash, 1.0f};

    auto subresource = vk::ImageSubresourceRange {
        vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0,
        vk::RemainingArrayLayers};

    mBackgroundDrawCallMgr.AddArgument_ClearColorImage(
        mRenderResMgr["DrawImage"]->GetTexHandle(), vk::ImageLayout::eGeneral,
        clearValue, {subresource});

    mBackgroundDrawCallMgr.AddArgument_Pipeline(
        vk::PipelineBindPoint::eCompute,
        mPipelineMgr.GetComputePipelineHandle("Background"));

    mBackgroundDrawCallMgr.AddArgument_DescriptorBuffer(
        {mDescMgr.GetDescBufferAddress(0)});

    auto offset =
        mDescMgr.GetDescriptorSet("Storage_Image_Buffer")->GetOffsetInBuffer();
    mBackgroundDrawCallMgr.AddArgument_DescriptorSet(
        vk::PipelineBindPoint::eCompute,
        mPipelineMgr.GetLayoutHandle("Background"), 0ui32, {0}, {offset});

    mBackgroundDrawCallMgr.AddArgument_Dispatch(
        ::std::ceil(mRenderResMgr["DrawImage"]->GetTexWidth() / 16.0),
        ::std::ceil(mRenderResMgr["DrawImage"]->GetTexHeight() / 16.0), 1);

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

void EngineCore::RecordDrawMeshCmds() {
    vk::ImageMemoryBarrier2 drawImageBarrier {
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eFragmentShader
            | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        {},
        mRenderResMgr["DrawImage"]->GetTexHandle(),
        Utils::GetWholeImageSubresource(vk::ImageAspectFlagBits::eColor)};
    mMeshDrawCallMgr.AddArgument_MemoryBarriers_BeforePass({"DrawImage"},
                                                           {drawImageBarrier});

    auto width = mRenderResMgr["DrawImage"]->GetTexWidth();
    auto height = mRenderResMgr["DrawImage"]->GetTexHeight();

    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment
        .setImageView(
            mRenderResMgr["DrawImage"]->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment
        .setImageView(
            mRenderResMgr["DepthImage"]->GetTexViewHandle("Depth-Whole"))
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    mMeshDrawCallMgr.AddArgument_RenderingInfo(
        {{0, 0}, {width, height}}, 1, 0, {colorAttachment}, depthAttachment);

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    mMeshDrawCallMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    mMeshDrawCallMgr.AddArgument_Scissor(0, {scissor});

    mMeshDrawCallMgr.AddArgument_Pipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetGraphicsPipelineHandle("TriangleDraw"));

    mMeshDrawCallMgr.AddArgument_DescriptorBuffer(
        {mDescMgr.GetDescBufferAddress(0)});

    auto sceneDataOffset =
        mDescMgr.GetDescriptorSet("Triangle_Scene_Data")->GetOffsetInBuffer();
    auto imageOffset =
        mDescMgr.GetDescriptorSet("ErrorCheck_Image")->GetOffsetInBuffer();
    mMeshDrawCallMgr.AddArgument_DescriptorSet(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetLayoutHandle("TriangleDraw"), 0ui32, {0, 0},
        {sceneDataOffset, imageOffset});

    mMeshDrawCallMgr.AddArgument_IndexBuffer(
        mFactoryModel->GetMeshBuffer().mIndexBuffer->GetBufferHandle(), 0,
        vk::IndexType::eUint32);

    auto pPushConstants = mFactoryModel->GetIndexDrawPushConstantsPtr();
    pPushConstants->mModelMatrix =
        glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});

    mMeshDrawCallMgr.AddArgument_PushConstant(
        mPipelineMgr.GetLayoutHandle("TriangleDraw"),
        vk::ShaderStageFlagBits::eVertex, 0, sizeof(*pPushConstants),
        pPushConstants);

    mMeshDrawCallMgr.AddArgument_DrawIndexedIndiret(
        mFactoryModel->GetIndexedIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawIndexedIndirectCommand));
}

void EngineCore::RecordDrawQuadCmds() {
    vk::ImageMemoryBarrier2 drawImageBarrier {
        vk::PipelineStageFlagBits2::eFragmentShader
            | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        {},
        {},
        mRenderResMgr["DrawImage"]->GetTexHandle(),
        Utils::GetWholeImageSubresource(vk::ImageAspectFlagBits::eColor)};

    auto scBarrier = mSwapchain->GetImageBarrier_BeforePass(
        mSwapchain->GetCurrentImageIndex());

    mQuadDrawCallMgr.AddArgument_MemoryBarriers_BeforePass(
        {"DrawImage", "Swapchain"}, {drawImageBarrier, scBarrier});

    auto width = mSwapchain->GetExtent2D().width;
    auto height = mSwapchain->GetExtent2D().height;

    auto imageIndex = mSwapchain->GetCurrentImageIndex();
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mSwapchain->GetImageViewHandle(imageIndex))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    mQuadDrawCallMgr.AddArgument_RenderingInfo({{0, 0}, {width, height}}, 1, 0,
                                               {colorAttachment});

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    mQuadDrawCallMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    mQuadDrawCallMgr.AddArgument_Scissor(0, {scissor});

    mQuadDrawCallMgr.AddArgument_Pipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetGraphicsPipelineHandle("QuadDraw"));

    mQuadDrawCallMgr.AddArgument_DescriptorBuffer(
        {mDescMgr.GetDescBufferAddress(0)});

    auto imageOffset =
        mDescMgr.GetDescriptorSet("DrawImage_Texture")->GetOffsetInBuffer();
    mQuadDrawCallMgr.AddArgument_DescriptorSet(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetLayoutHandle("QuadDraw"), 0ui32, {0}, {imageOffset});

    mQuadDrawCallMgr.AddArgument_Draw(3, 1, 0, 0);

    scBarrier = mSwapchain->GetImageBarrier_AfterPass(
        mSwapchain->GetCurrentImageIndex());

    mQuadDrawCallMgr.AddArgument_MemoryBarriers_AfterPass({"Swapchain"},
                                                          {scBarrier});
}

void EngineCore::RecordMeshShaderDrawCmds() {
    vk::ImageMemoryBarrier2 drawImageBarrier {
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eFragmentShader
            | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        {},
        mRenderResMgr["DrawImage"]->GetTexHandle(),
        Utils::GetWholeImageSubresource(vk::ImageAspectFlagBits::eColor)};
    mMeshShaderDrawCallMgr.AddArgument_MemoryBarriers_BeforePass(
        {"DrawImage"}, {drawImageBarrier});

    auto width = mRenderResMgr["DrawImage"]->GetTexWidth();
    auto height = mRenderResMgr["DrawImage"]->GetTexHeight();

    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment
        .setImageView(
            mRenderResMgr["DrawImage"]->GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingAttachmentInfo depthAttachment {};
    depthAttachment
        .setImageView(
            mRenderResMgr["DepthImage"]->GetTexViewHandle("Depth-Whole"))
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue {0.0f});

    mMeshShaderDrawCallMgr.AddArgument_RenderingInfo(
        {{0, 0}, {width, height}}, 1, 0, {colorAttachment}, depthAttachment);

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    mMeshShaderDrawCallMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    mMeshShaderDrawCallMgr.AddArgument_Scissor(0, {scissor});

    mMeshShaderDrawCallMgr.AddArgument_Pipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetGraphicsPipelineHandle("MeshShaderDraw"));

    mMeshShaderDrawCallMgr.AddArgument_DescriptorBuffer(
        {mDescMgr.GetDescBufferAddress(0)});

    auto sceneDataOffset =
        mDescMgr.GetDescriptorSet("MeshShader_Scene_Data")->GetOffsetInBuffer();
    mMeshShaderDrawCallMgr.AddArgument_DescriptorSet(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetLayoutHandle("MeshShaderDraw"), 0ui32, {0},
        {sceneDataOffset});

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    meshPushContants->mModelMatrix =
        glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});

    mMeshShaderDrawCallMgr.AddArgument_PushConstant(
        mPipelineMgr.GetLayoutHandle("MeshShaderDraw"),
        vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(*meshPushContants),
        meshPushContants);

    mMeshShaderDrawCallMgr.AddArgument_DrawMeshTasksIndirect(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));
}

}  // namespace IntelliDesign_NS::Vulkan::Core