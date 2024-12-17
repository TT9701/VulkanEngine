#include "MeshShaderDemo.h"

#include <random>

using namespace IDNS_VC;

MeshShaderDemo::MeshShaderDemo(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescSetPool(GetVulkanContext())),
      mPrepassCopy(CreateBindingInfo_Copy()),
      mBackgroundPass_PSO {CreateBindingInfo_PSO()},
      mBackgroundPass_Barrier(CreateBindingInfo_Barrier()),
      mMeshDrawPass_PSO {CreateBindingInfo_PSO()},
      mMeshDrawPass_Barrier(CreateBindingInfo_Barrier()),
      mMeshShaderPass_PSO {CreateBindingInfo_PSO()},
      mMeshShaderPass_Barrier_Pre(CreateBindingInfo_Barrier()),
      mMeshShaderPass_Barrier_Post(CreateBindingInfo_Barrier()),
      mQuadDrawPass_PSO {CreateBindingInfo_PSO(true)},
      mQuadDrawPass_Barrier_Pre(CreateBindingInfo_Barrier(true)),
      mQuadDrawPass_Barrier_Post(CreateBindingInfo_Barrier(true)),
      mShadowPass_PSO {CreateBindingInfo_PSO()},
      mShadowPass_Barrier(CreateBindingInfo_Barrier()),
      mRuntimeCopy {&GetRenderResMgr()},
      mRuntimeCopy_Barrier_Pre(CreateBindingInfo_Barrier()),
      mRuntimeCopy_Barrier_Post(CreateBindingInfo_Barrier()),
      mCopySem(GetVulkanContext()),
      mCmpSem(GetVulkanContext()),
      mGui(GetVulkanContext(), GetSwapchain(), GetSDLWindow()) {}

MeshShaderDemo::~MeshShaderDemo() = default;

void MeshShaderDemo::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
}

void MeshShaderDemo::LoadShaders() {
    auto& shaderMgr = GetShaderMgr();

    shaderMgr.CreateShaderFromGLSL("computeDraw",
                                   SHADER_PATH_CSTR("BackGround.comp"),
                                   vk::ShaderStageFlagBits::eCompute);

    shaderMgr.CreateShaderFromGLSL("vertex", SHADER_PATH_CSTR("Triangle.vert"),
                                   vk::ShaderStageFlagBits::eVertex, true);

    shaderMgr.CreateShaderFromGLSL("fragment",
                                   SHADER_PATH_CSTR("Triangle.frag"),
                                   vk::ShaderStageFlagBits::eFragment, true);

    shaderMgr.CreateShaderFromGLSL("Mesh shader fragment",
                                   SHADER_PATH_CSTR("MeshShader.frag"),
                                   vk::ShaderStageFlagBits::eFragment, true);

    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    shaderMgr.CreateShaderFromGLSL(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        vk::ShaderStageFlagBits::eTaskEXT, false, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    shaderMgr.CreateShaderFromGLSL(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        vk::ShaderStageFlagBits::eMeshEXT, true, macros);

    shaderMgr.CreateShaderFromGLSL("Quad vertex", SHADER_PATH_CSTR("Quad.vert"),
                                   vk::ShaderStageFlagBits::eVertex);

    shaderMgr.CreateShaderFromGLSL("Quad fragment",
                                   SHADER_PATH_CSTR("Quad.frag"),
                                   vk::ShaderStageFlagBits::eFragment);
}

void MeshShaderDemo::PollEvents(SDL_Event* e, float deltaTime) {
    Application::PollEvents(e, deltaTime);

    mMainCamera.ProcessSDLEvent(e, deltaTime);
    mGui.PollEvent(e);
}

void MeshShaderDemo::Update_OnResize() {
    Application::Update_OnResize();

    auto& window = GetSDLWindow();
    auto& renderResMgr = GetRenderResMgr();

    vk::Extent2D extent = {static_cast<uint32_t>(window.GetWidth()),
                           static_cast<uint32_t>(window.GetHeight())};

    renderResMgr.ResizeResources_ScreenSizeRelated(extent);

    auto resNames = renderResMgr.GetResourceNames_SrcreenSizeRelated();

    mBackgroundPass_PSO.Update(resNames);
    mBackgroundPass_Barrier.Update(resNames);

    // mMeshDrawPass_PSO.OnResize(extent);
    // mMeshDrawPass_Barrier.OnResize(extent);

    mMeshShaderPass_PSO.OnResize(extent);
    mMeshShaderPass_Barrier_Pre.Update(resNames);

    mQuadDrawPass_Barrier_Pre.Update(resNames);
    mQuadDrawPass_PSO.OnResize(extent);
    mQuadDrawPass_Barrier_Post.Update(resNames);
}

void MeshShaderDemo::UpdateScene() {
    Application::UpdateScene();

    auto& window = GetSDLWindow();

    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(window.GetWidth())
                             / static_cast<float>(window.GetHeight()),
                         500.0f, 0.01f);

    proj[1][1] *= -1;

    mSceneData.cameraPos = glm::vec4 {mMainCamera.mPosition, 1.0f};
    mSceneData.view = view;
    mSceneData.proj = proj;
    mSceneData.viewProj = proj * view;
    UpdateSceneUBO();
}

void MeshShaderDemo::Prepare() {
    Application::Prepare();

    auto& window = GetSDLWindow();

    for (auto& frame : GetFrames()) {
        frame.PrepareBindlessDescPool({&mMeshShaderPass_PSO});
    }

    CreateRandomTexture();

    CreateDrawImage();
    CreateDepthImage();

    CreateShadowImages();

    LoadShaders();
    CreatePipelines();

    auto& renderResMgr = GetRenderResMgr();

    renderResMgr.CreateBuffer(
        "SceneUniformBuffer", sizeof(SceneData),
        vk::BufferUsageFlagBits::eUniformBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::Staging);

    renderResMgr.CreateBuffer_ScreenSizeRelated(
        "RWBuffer", sizeof(glm::vec4) * window.GetWidth() * window.GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::DeviceLocal, sizeof(glm::vec4));

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // cisdi model data converter
    {
        IntelliDesign_NS::Core::Utils::Timer timer;

        Type_STLString model = "sponza/sponza.obj";

        mFactoryModel = MakeShared<Geometry>(MODEL_PATH_CSTR(model));

        // mFactoryModel->GenerateBuffers(&GetVulkanContext(), this);
        mFactoryModel->GenerateMeshletBuffers(&GetVulkanContext(), this);

        auto duration_LoadModel = timer.End();
        printf("Load Geometry: %s, Time consumed: %f s. \n", model.c_str(),
               duration_LoadModel);
    }

    // {
    //     const uint32_t modelCount = 30;
    //     mModels.resize(modelCount);
    //     IntelliDesign_NS::Core::Utils::Timer timer;
    //
    //     Type_STLString model = "equipment/Model";
    //     Type_STLString postfix = ".stl";
    //
    //     for (uint32_t i = 0; i < modelCount; ++i) {
    //         auto name = model + std::to_string(i).c_str() + postfix;
    //         // IntelliDesign_NS::ModelData::CISDI_3DModel::Convert(
    //         //     MODEL_PATH_CSTR(name), true);
    //
    //         auto cisdiModel = IntelliDesign_NS::ModelData::CISDI_3DModel::Load(
    //             (MODEL_PATH(name) + CISDI_3DModel_Subfix_Str).c_str());
    //
    //         mModels[i] = MakeShared<Geometry>(::std::move(cisdiModel));
    //         mModels[i]->GenerateMeshletBuffers(&GetVulkanContext(), this);
    //     }
    //
    //     auto duration_LoadModel = timer.End();
    //     printf("Time consumed: %f s. \n", duration_LoadModel);
    // }

    RecordDrawBackgroundCmds();
    // RecordDrawMeshCmds();
    RecordMeshShaderDrawCmds();
    RecordDrawQuadCmds();
    RecordShadowDrawCmds();
    RecordRuntimeCopyCmds();

    PrepareUIContext();
}

void MeshShaderDemo::BeginFrame(IDNS_VC::RenderFrame& frame) {
    Application::BeginFrame(frame);
    mGui.BeginFrame();
}

void MeshShaderDemo::RenderFrame(IDNS_VC::RenderFrame& frame) {
    auto& vkCtx = GetVulkanContext();
    auto& timelineSem = vkCtx.GetTimelineSemphore();
    auto& cmdMgr = GetCmdMgr();

    const uint64_t graphicsFinished = timelineSem.GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t copyFinished = graphicsFinished + 2;
    const uint64_t shadowFinished = graphicsFinished + 3;
    const uint64_t allFinished = graphicsFinished + 4;

    // Compute Draw
    {
        auto cmd = frame.GetComputeCmdBuf();

        mBackgroundPass_Barrier.RecordCmd(cmd.GetHandle());
        mBackgroundPass_PSO.RecordCmd(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eBottomOfPipe, timelineSem.GetHandle(),
             graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllCommands, mCmpSem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eAllCommands, timelineSem.GetHandle(),
             computeFinished}};

        cmdMgr.Submit(cmd.GetHandle(),
                      vkCtx.GetQueue(QueueUsage::Compute_Runtime).GetHandle(),
                      waits, signals, frame.GetFencePool().RequestFence());
    }

    // Runtime Copy
    {
        auto cmd = frame.GetTransferCmdBuf();

        mRuntimeCopy_Barrier_Pre.RecordCmd(cmd.GetHandle());
        mRuntimeCopy.RecordCmd(cmd.GetHandle());
        mRuntimeCopy_Barrier_Post.RecordCmd(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             frame.GetReady4RenderSemaphore().GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eBottomOfPipe, timelineSem.GetHandle(),
             graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllCommands, mCopySem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eAllCommands, timelineSem.GetHandle(),
             copyFinished}};

        cmdMgr.Submit(
            cmd.GetHandle(),
            vkCtx.GetQueue(QueueUsage::Transfer_Runtime_Upload).GetHandle(),
            waits, signals, frame.GetFencePool().RequestFence());
    }

    // Shadow Draw
    {
        auto cmd = frame.GetGraphicsCmdBuf();

        mShadowPass_Barrier.RecordCmd(cmd.GetHandle());
        for (uint32_t i = 0; i < 12; ++i) {
            mShadowPass_PSO.RecordCmd(cmd.GetHandle());
        }

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eBottomOfPipe, timelineSem.GetHandle(),
             graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllCommands, timelineSem.GetHandle(),
             shadowFinished}};

        cmdMgr.Submit(cmd.GetHandle(),
                      vkCtx.GetQueue(QueueUsage::Graphics).GetHandle(), waits,
                      signals, frame.GetFencePool().RequestFence());
    }

    // Graphics Draw
    {
        auto cmd = frame.GetGraphicsCmdBuf();

        // mMeshDrawPass_Barrier.RecordCmd(cmd.GetHandle());
        // mMeshDrawPass_PSO.RecordCmd(cmd.GetHandle());

        mMeshShaderPass_Barrier_Pre.RecordCmd(cmd.GetHandle());
        mMeshShaderPass_PSO.RecordCmd(cmd.GetHandle());
        mMeshShaderPass_Barrier_Post.RecordCmd(cmd.GetHandle());

        mQuadDrawPass_Barrier_Pre.Update("_Swapchain_");
        mQuadDrawPass_PSO.Update("_Swapchain_");
        mQuadDrawPass_Barrier_Post.Update("_Swapchain_");

        mQuadDrawPass_Barrier_Pre.RecordCmd(cmd.GetHandle());
        mQuadDrawPass_PSO.RecordCmd(cmd.GetHandle());
        mQuadDrawPass_Barrier_Post.RecordCmd(cmd.GetHandle());

        mGui.Draw(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eAllCommands, mCmpSem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eAllCommands, mCopySem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eComputeShader,
             timelineSem.GetHandle(), shadowFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics, timelineSem.GetHandle(),
             allFinished},
            {vk::PipelineStageFlagBits2::eAllGraphics,
             frame.GetReady4PresentSemaphore().GetHandle()}};

        cmdMgr.Submit(cmd.GetHandle(),
                      vkCtx.GetQueue(QueueUsage::Graphics).GetHandle(), waits,
                      signals, frame.GetFencePool().RequestFence());
    }
}

void MeshShaderDemo::EndFrame(IDNS_VC::RenderFrame& frame) {
    Application::EndFrame(frame);
}

void MeshShaderDemo::CreateDrawImage() {
    auto& window = GetSDLWindow();

    vk::Extent3D drawImageExtent {static_cast<uint32_t>(window.GetWidth()),
                                  static_cast<uint32_t>(window.GetHeight()), 1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto& ref = GetRenderResMgr().CreateTexture_ScreenSizeRelated(
        "DrawImage", RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, drawImageExtent, drawImageUsage);
    ref.CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
}

void MeshShaderDemo::CreateDepthImage() {
    auto& window = GetSDLWindow();

    vk::Extent3D depthImageExtent {static_cast<uint32_t>(window.GetWidth()),
                                   static_cast<uint32_t>(window.GetHeight()),
                                   1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto& ref = GetRenderResMgr().CreateTexture_ScreenSizeRelated(
        "DepthImage", RenderResource::Type::Texture2D,
        vk::Format::eD24UnormS8Uint, depthImageExtent, depthImageUsage);
    ref.CreateTexView("Depth-Whole", vk::ImageAspectFlagBits::eDepth
                                         | vk::ImageAspectFlagBits::eStencil);

    auto& vkCtx = GetVulkanContext();
    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueUsage::Graphics));
        Utils::TransitionImageLayout(
            cmd.mHandle, ref.GetTexHandle(), vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }
}

void MeshShaderDemo::CreateRandomTexture() {
    constexpr auto extent = vk::Extent3D {512, 512, 1};
    constexpr uint32_t randomImageCount = 64;

    constexpr size_t dataSize = extent.width * extent.height * 4;

    auto& renderResMgr = GetRenderResMgr();

    auto& uploadBuffer = renderResMgr.CreateBuffer(
        "staging", dataSize * randomImageCount,
        vk::BufferUsageFlagBits::eTransferSrc, Buffer::MemoryType::Staging);

    ::std::string baseName {"RandomImage"};

    for (uint32_t i = 0; i < randomImageCount; ++i) {
        std::random_device rndDevice;
        std::default_random_engine rndEngine(rndDevice());
        std::uniform_int_distribution rndDist(50, UCHAR_MAX);

        std::vector<uint8_t> pixels(dataSize);
        for (uint32_t j = 0; j < extent.width * extent.height; ++j) {
            pixels[j * 4] = rndDist(rndEngine);
            pixels[j * 4 + 1] = rndDist(rndEngine);
            pixels[j * 4 + 2] = rndDist(rndEngine);
            pixels[j * 4 + 3] = 255;
        }

        auto name = baseName + ::std::to_string(i);
        auto& ref = renderResMgr.CreateTexture(
            name.c_str(), RenderResource::Type::Texture2D,
            vk::Format::eR8G8B8A8Unorm, extent,
            vk::ImageUsageFlagBits::eSampled
                | vk::ImageUsageFlagBits::eTransferDst);
        ref.CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

        memcpy((char*)uploadBuffer.GetBufferMappedPtr() + dataSize * i,
               pixels.data(), dataSize);
    }

    auto& vkCtx = GetVulkanContext();
    auto& device = vkCtx.GetDevice();

    auto gfxQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics);
    auto tsfQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer);

    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueUsage::Graphics));

        for (uint32_t i = 0; i < randomImageCount; ++i) {
            auto name = baseName + ::std::to_string(i);
            Utils::TransitionImageLayout(
                cmd.mHandle, renderResMgr[name.c_str()].GetTexHandle(),
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal);

            vk::BufferImageCopy2 copyRegion {};
            copyRegion.setBufferOffset(dataSize * i)
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(extent);

            mPrepassCopy.CopyBufferToImage("staging", name.c_str(), copyRegion);
        }

        mPrepassCopy.GenerateMetaData();

        mPrepassCopy.RecordCmd(cmd.mHandle);

        for (uint32_t i = 0; i < randomImageCount; ++i) {
            auto name = baseName + ::std::to_string(i);

            Utils::TransitionImageLayout(
                cmd.mHandle, renderResMgr[name.c_str()].GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        for (uint32_t i = 0; i < 16; ++i) {
            auto name = baseName + ::std::to_string(i);
            vk::ImageMemoryBarrier2 barrier2 {};
            barrier2.setImage(renderResMgr[name.c_str()].GetTexHandle())
                .setSrcQueueFamilyIndex(gfxQueueIdx)
                .setDstQueueFamilyIndex(tsfQueueIdx)
                .setSubresourceRange(Utils::GetWholeImageSubresource(
                    vk::ImageAspectFlagBits::eColor));

            vk::DependencyInfo dep {};
            dep.setImageMemoryBarriers(barrier2);
            cmd->pipelineBarrier2(dep);
        }
    }

    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        for (uint32_t j = 0; j < randomImageCount / 2; ++j) {
            auto name = baseName + ::std::to_string(j);
            auto texture = renderResMgr[name.c_str()].GetTexPtr();

            GetCurFrame().GetBindlessDescPool().Add(texture);
        }
    }
}

void MeshShaderDemo::CreateShadowImages() {
    vk::Extent3D extent {1024, 1024, 1};

    auto& renderResMgr = GetRenderResMgr();

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto& shadowColor = renderResMgr.CreateTexture_ScreenSizeRelated(
        "ShadowColor", RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, extent, drawImageUsage);
    shadowColor.CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto& shadowDepth = renderResMgr.CreateTexture_ScreenSizeRelated(
        "ShadowDepth", RenderResource::Type::Texture2D,
        vk::Format::eD24UnormS8Uint, extent, depthImageUsage);
    shadowDepth.CreateTexView(
        "Depth-Whole",
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);

    auto& vkCtx = GetVulkanContext();
    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueUsage::Graphics));
        Utils::TransitionImageLayout(
            cmd.mHandle, shadowDepth.GetTexHandle(),
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }
}

void MeshShaderDemo::CreateBackgroundComputePipeline() {
    auto& shaderMgr = GetShaderMgr();

    auto compute =
        shaderMgr.GetShader("computeDraw", vk::ShaderStageFlagBits::eCompute);
    auto program = shaderMgr.CreateProgram("background", compute);

    auto builder = GetPipelineMgr().GetBuilder_Compute();

    auto backgroundComputePipeline =
        builder.SetShaderProgram(program)
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void MeshShaderDemo::CreateMeshPipeline() {
    auto& renderResMgr = GetRenderResMgr();
    auto& shaderMgr = GetShaderMgr();

    auto vert = shaderMgr.GetShader("vertex", vk::ShaderStageFlagBits::eVertex);
    auto frag =
        shaderMgr.GetShader("fragment", vk::ShaderStageFlagBits::eFragment);
    auto program = shaderMgr.CreateProgram("mesh draw", vert, frag);

    auto builder = GetPipelineMgr().GetBuilder_Graphics();
    builder.SetShaderProgram(program)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eFront,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(renderResMgr["DrawImage"].GetTexFormat())
        .SetDepthStencilFormat(renderResMgr["DepthImage"].GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("TriangleDraw");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void MeshShaderDemo::CreateMeshShaderPipeline() {
    auto& renderResMgr = GetRenderResMgr();
    auto& shaderMgr = GetShaderMgr();

    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));

    auto task = shaderMgr.GetShader("Mesh shader task",
                                    vk::ShaderStageFlagBits::eTaskEXT, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));

    auto mesh = shaderMgr.GetShader("Mesh shader mesh",
                                    vk::ShaderStageFlagBits::eMeshEXT, macros);

    auto frag = shaderMgr.GetShader("Mesh shader fragment",
                                    vk::ShaderStageFlagBits::eFragment);

    auto program = shaderMgr.CreateProgram("meshlet draw", task, mesh, frag);

    auto builder = GetPipelineMgr().GetBuilder_Graphics();
    builder.SetShaderProgram(program)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(renderResMgr["DrawImage"].GetTexFormat())
        .SetDepthStencilFormat(renderResMgr["DepthImage"].GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("MeshShaderDraw");

    DBG_LOG_INFO("Vulkan MeshShader Graphics Pipeline Created");
}

void MeshShaderDemo::CreateDrawQuadPipeline() {
    auto& shaderMgr = GetShaderMgr();

    auto vert =
        shaderMgr.GetShader("Quad vertex", vk::ShaderStageFlagBits::eVertex);
    auto frag = shaderMgr.GetShader("Quad fragment",
                                    vk::ShaderStageFlagBits::eFragment);

    auto program = shaderMgr.CreateProgram("QuadDraw", vert, frag);

    auto builder = GetPipelineMgr().GetBuilder_Graphics();
    builder.SetShaderProgram(program)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::False, vk::False)
        .SetColorAttachmentFormat(GetSwapchain().GetFormat())
        .SetDepthStencilFormat(vk::Format::eUndefined)
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("QuadDraw");

    DBG_LOG_INFO("Vulkan Quad Graphics Pipeline Created");
}

void MeshShaderDemo::RecordDrawBackgroundCmds() {
    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    // barrier
    {
        mBackgroundPass_Barrier.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .newLayout = vk::ImageLayout::eGeneral,
             .aspect = vk::ImageAspectFlagBits::eColor});

        mBackgroundPass_Barrier.AddImageBarrier(
            "ShadowColor",
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .newLayout = vk::ImageLayout::eGeneral,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mBackgroundPass_Barrier.GenerateMetaData();
    }

    // PSO
    {
        mBackgroundPass_PSO.SetPipeline("Background");

        mBackgroundPass_PSO["image"] = "DrawImage";
        mBackgroundPass_PSO["StorageBuffer"] = "RWBuffer";

        mBackgroundPass_PSO.GenerateMetaData();
    }

    auto& dcMgr = mBackgroundPass_PSO.GetDrawCallManager();

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));
    vk::ClearColorValue clearValue {flash, flash, flash, 1.0f};
    auto subresource = vk::ImageSubresourceRange {
        vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0,
        vk::RemainingArrayLayers};
    // dcMgr.AddArgument_ClearColorImage("DrawImage", vk::ImageLayout::eGeneral,
    //                                   clearValue, {subresource});

    dcMgr.AddArgument_Dispatch(::std::ceil(width / 16.0),
                               ::std::ceil(height / 16.0), 1);
}

void MeshShaderDemo::RecordDrawMeshCmds() {
    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    auto pPushConstants = mFactoryModel->GetIndexDrawPushConstantsPtr();
    pPushConstants->mModelMatrix =
        glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});

    // barrier
    {
        mMeshDrawPass_Barrier.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader
                           | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .oldLayout = vk::ImageLayout::eGeneral,
             .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mMeshDrawPass_Barrier.GenerateMetaData();
    }

    // PSO
    {
        mMeshDrawPass_PSO.SetPipeline("TriangleDraw");

        mMeshDrawPass_PSO["constants"] = RenderPassBinding::PushContants {
            sizeof(*pPushConstants), pPushConstants};

        mMeshDrawPass_PSO["SceneDataUBO"] = "SceneUniformBuffer";
        mMeshDrawPass_PSO["tex"] = "ErrorCheckImage";

        mMeshDrawPass_PSO["outFragColor"] = {"DrawImage", "Color-Whole"};

        mMeshDrawPass_PSO[RenderPassBinding::Type::DSV] = {"DepthImage",
                                                           "Depth-Whole"};

        mMeshDrawPass_PSO[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mMeshDrawPass_PSO.GenerateMetaData();
    }

    auto& dcMgr = mMeshDrawPass_PSO.GetDrawCallManager();

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_DrawIndiret(
        mFactoryModel->GetIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshCount(), sizeof(vk::DrawIndirectCommand));
}

void MeshShaderDemo::RecordDrawQuadCmds() {
    auto extent = GetSwapchain().GetExtent2D();

    // barriers
    {
        mQuadDrawPass_Barrier_Pre.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader
                           | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
             .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});

        mQuadDrawPass_Barrier_Pre.AddImageBarrier(
            "_Swapchain_",
            {.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});

        mQuadDrawPass_Barrier_Pre.GenerateMetaData();

        mQuadDrawPass_Barrier_Post.AddImageBarrier(
            "_Swapchain_",
            {.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
             .dstAccessMask = vk::AccessFlagBits2::eNone,
             .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .newLayout = vk::ImageLayout::ePresentSrcKHR,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mQuadDrawPass_Barrier_Post.GenerateMetaData();
    }

    // PSO
    {
        mQuadDrawPass_PSO.SetPipeline("QuadDraw");

        mQuadDrawPass_PSO["tex"] = "DrawImage";

        mQuadDrawPass_PSO["outFragColor"] = {"_Swapchain_", ""};

        mQuadDrawPass_PSO[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {
                {{0, 0}, {extent.width, extent.height}}, 1, 0};

        mQuadDrawPass_PSO.GenerateMetaData();
    }
    auto& dcMgr = mQuadDrawPass_PSO.GetDrawCallManager();

    vk::Viewport viewport {
        0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {extent.width, extent.height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_Draw(3, 1, 0, 0);
}

void MeshShaderDemo::RecordMeshShaderDrawCmds() {
    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    // meshPushContants->mModelMatrix =
    //     glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});
    meshPushContants->mModelMatrix =
        glm::rotate(glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.01f}),
                    glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

    uint32_t copyCount = 16;
    ::std::string baseName {"RandomImage"};

    auto& device = GetVulkanContext().GetDevice();

    auto gfxQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics);
    auto tsfQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer);

    // barrier
    {
        for (uint32_t i = 0; i < copyCount; ++i) {
            auto name = baseName + ::std::to_string(i);
            mMeshShaderPass_Barrier_Pre.AddImageBarrier(
                name.c_str(),
                {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
                 .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .srcQueueFamilyIndex = tsfQueueIdx,
                 .dstQueueFamilyIndex = gfxQueueIdx,
                 .aspect = vk::ImageAspectFlagBits::eColor});
        }

        mMeshShaderPass_Barrier_Pre.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader
                           | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .oldLayout = vk::ImageLayout::eGeneral,
             .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mMeshShaderPass_Barrier_Pre.GenerateMetaData();
    }

    // PSO
    {
        mMeshShaderPass_PSO.SetPipeline("MeshShaderDraw");

        mMeshShaderPass_PSO["PushConstants"] = RenderPassBinding::PushContants {
            sizeof(*meshPushContants), meshPushContants};

        mMeshShaderPass_PSO["UBO"] = "SceneUniformBuffer";

        auto bindlessSet =
            GetCurFrame().GetBindlessDescPool().GetPoolResource();
        mMeshShaderPass_PSO["sceneTexs"] =
            RenderPassBinding::BindlessDescBufInfo {bindlessSet.deviceAddr,
                                                    bindlessSet.offset};

        mMeshShaderPass_PSO["outFragColor"] = {"DrawImage", "Color-Whole"};
        mMeshShaderPass_PSO[RenderPassBinding::Type::DSV] = {"DepthImage",
                                                             "Depth-Whole"};

        mMeshShaderPass_PSO[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mMeshShaderPass_PSO.GenerateMetaData();
    }

    auto& dcMgr = mMeshShaderPass_PSO.GetDrawCallManager();

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_DrawMeshTasksIndirect(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshCount(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));

    // for (auto const& model : mModels) {
    //     dcMgr.AddArgument_DrawMeshTasksIndirect(
    //         model->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
    //         model->GetMeshCount(), sizeof(vk::DrawMeshTasksIndirectCommandEXT));
    // }

    // barrier
    {
        for (uint32_t i = 0; i < copyCount; ++i) {
            auto name = baseName + ::std::to_string(i);
            mMeshShaderPass_Barrier_Post.AddImageBarrier(
                name.c_str(),
                {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                 .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .newLayout = vk::ImageLayout::eTransferDstOptimal,
                 .srcQueueFamilyIndex = gfxQueueIdx,
                 .dstQueueFamilyIndex = tsfQueueIdx,
                 .aspect = vk::ImageAspectFlagBits::eColor});
        }

        mMeshShaderPass_Barrier_Post.GenerateMetaData();
    }
}

void MeshShaderDemo::RecordShadowDrawCmds() {
    auto& shadowColor = GetRenderResMgr()["ShadowColor"];
    uint32_t width = shadowColor.GetTexWidth();
    uint32_t height = shadowColor.GetTexHeight();

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    // meshPushContants->mModelMatrix =
    //     glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});
    meshPushContants->mModelMatrix =
        glm::rotate(glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.01f}),
                    glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

    // barrier
    {
        mShadowPass_Barrier.AddImageBarrier(
            "ShadowColor",
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader
                           | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .oldLayout = vk::ImageLayout::eGeneral,
             .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mShadowPass_Barrier.GenerateMetaData();
    }

    // PSO
    {
        mShadowPass_PSO.SetPipeline("MeshShaderDraw");

        mShadowPass_PSO["PushConstants"] = RenderPassBinding::PushContants {
            sizeof(*meshPushContants), meshPushContants};

        mShadowPass_PSO["UBO"] = "SceneUniformBuffer";

        auto bindlessSet =
            GetCurFrame().GetBindlessDescPool().GetPoolResource();
        mShadowPass_PSO["sceneTexs"] = RenderPassBinding::BindlessDescBufInfo {
            bindlessSet.deviceAddr, bindlessSet.offset};

        mShadowPass_PSO["outFragColor"] = {"ShadowColor", "Color-Whole"};
        mShadowPass_PSO[RenderPassBinding::Type::DSV] = {"ShadowDepth",
                                                         "Depth-Whole"};

        mShadowPass_PSO[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mShadowPass_PSO.GenerateMetaData();
    }

    auto& dcMgr = mShadowPass_PSO.GetDrawCallManager();

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_DrawMeshTasksIndirect(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshCount(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));
}

void MeshShaderDemo::RecordRuntimeCopyCmds() {
    uint32_t copyCount = 16;
    constexpr auto extent = vk::Extent3D {512, 512, 1};
    constexpr size_t dataSize = extent.width * extent.height * 4;
    ::std::string baseName {"RandomImage"};

    auto& device = GetVulkanContext().GetDevice();

    auto gfxQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics);
    auto tsfQueueIdx = device.GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer);

    for (uint32_t i = 0; i < copyCount; ++i) {
        auto name = baseName + ::std::to_string(i);
        mRuntimeCopy_Barrier_Pre.AddImageBarrier(
            name.c_str(),
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite
                            | vk::AccessFlagBits2::eMemoryRead,
             .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
             .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
             .newLayout = vk::ImageLayout::eTransferDstOptimal,
             .srcQueueFamilyIndex = gfxQueueIdx,
             .dstQueueFamilyIndex = tsfQueueIdx,
             .aspect = vk::ImageAspectFlagBits::eColor});
    }
    mRuntimeCopy_Barrier_Pre.GenerateMetaData();

    for (uint32_t i = 0; i < copyCount; ++i) {
        auto name = baseName + ::std::to_string(i);

        vk::BufferImageCopy2 copyRegion {};
        copyRegion.setBufferOffset(dataSize * i)
            .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(extent);

        mRuntimeCopy.CopyBufferToImage("staging", name.c_str(), copyRegion);
    }
    mRuntimeCopy.GenerateMetaData();

    for (uint32_t i = 0; i < copyCount; ++i) {
        auto name = baseName + ::std::to_string(i);
        mRuntimeCopy_Barrier_Post.AddImageBarrier(
            name.c_str(),
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
             .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
             .oldLayout = vk::ImageLayout::eTransferDstOptimal,
             .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
             .srcQueueFamilyIndex = tsfQueueIdx,
             .dstQueueFamilyIndex = gfxQueueIdx,
             .aspect = vk::ImageAspectFlagBits::eColor});
    }
    mRuntimeCopy_Barrier_Post.GenerateMetaData();
}

void MeshShaderDemo::UpdateSceneUBO() {
    auto& renderResMgr = GetRenderResMgr();
    auto data = renderResMgr["SceneUniformBuffer"].GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

void MeshShaderDemo::PrepareUIContext() {
    mImageName0 = "RandomImage";
    mImageName1 = "RandomImage";
    auto& renderResMgr = GetRenderResMgr();
    mGui.AddContext([&]() {
        if (ImGui::Begin("SceneStats")) {
            ImGui::Text("Camera Position: (%.3f, %.3f, %.3f)",
                        mSceneData.cameraPos.x, mSceneData.cameraPos.y,
                        mSceneData.cameraPos.z);
            ImGui::SliderFloat3("Sun light position",
                                (float*)&mSceneData.sunLightPos, -1.0f, 1.0f);
            ImGui::ColorEdit4("ObjColor", (float*)&mSceneData.objColor);
            ImGui::SliderFloat2("MetallicRoughness",
                                (float*)&mSceneData.metallicRoughness, 0.0f,
                                1.0f);
            ImGui::InputInt("Texture Index", &mSceneData.texIndex);

            ImGui::InputText("##", mImageName0.data(), 32);
            ImGui::SameLine();
            if (ImGui::Button("Add")) {
                auto tex = renderResMgr[mImageName0.c_str()].GetTexPtr();
                auto idx = GetCurFrame().GetBindlessDescPool().Add(tex);

                DBG_LOG_INFO("Add Button pressed, add texture at idx %d", idx);
            }

            ImGui::InputText("###", mImageName1.data(), 32);
            ImGui::SameLine();
            if (ImGui::Button("Delete")) {
                auto tex = renderResMgr[mImageName1.c_str()].GetTexPtr();
                auto idx = GetCurFrame().GetBindlessDescPool().Delete(tex);
                DBG_LOG_INFO("Delete Button pressed, delete texture at idx %d",
                             idx);
            }
        }
        ImGui::End();
    });
}

IDNS_VC::RenderPassBindingInfo_PSO MeshShaderDemo::CreateBindingInfo_PSO(
    bool swapchain) {
    Swapchain* p = nullptr;
    if (swapchain)
        p = &GetSwapchain();
    return {&GetVulkanContext(), &GetRenderResMgr(), &GetPipelineMgr(),
            &mDescSetPool, p};
}

IDNS_VC::RenderPassBindingInfo_Barrier
MeshShaderDemo::CreateBindingInfo_Barrier(bool swapchain) {
    Swapchain* p = nullptr;
    if (swapchain)
        p = &GetSwapchain();
    return {&GetVulkanContext(), &GetRenderResMgr(), p};
}

IDNS_VC::RenderPassBindingInfo_Copy MeshShaderDemo::CreateBindingInfo_Copy() {
    return {&GetRenderResMgr()};
}