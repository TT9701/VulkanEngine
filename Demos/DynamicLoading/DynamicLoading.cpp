#include "DynamicLoading.h"

#include "Core/System/GameTimer.h"
#include "Core/System/MemoryPool/MemoryPool.h"

using namespace IDVC_NS;

namespace {
::std::pmr::memory_resource* gMemPool = ::std::pmr::get_default_resource();

void AdjustCameraPosition(
    IDC_NS::Camera& camera,
    IntelliDesign_NS::ModelData::AABoundingBox const& bb) {
    auto center = bb.Center.GetSIMD();
    auto extent = bb.Extents.GetSIMD();
    camera.AdjustPosition(center, extent);
}

}  // namespace

DynamicLoading::DynamicLoading(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescSetPool(GetVulkanContext())),
      mRenderSequence(GetVulkanContext(), GetRenderResMgr(), GetPipelineMgr(),
                      mDescSetPool),
      mCopySem(GetVulkanContext()),
      mCmpSem(GetVulkanContext()) {
    mMainCamera = MakeUnique<IDC_NS::Camera>(IDC_NS::PersperctiveInfo {
        50000.0f, 0.01f, IDCMCore_NS::ConvertToRadians(45.0f),
        (float)spec.width / spec.height});

    mModelMgr =
        MakeUnique<IntelliDesign_NS::ModelData::ModelDataManager>(gMemPool);

    mGeoMgr = MakeUnique<GPUGeometryDataManager>(GetVulkanContext(),
                                                 DGC_MAX_DRAW_COUNT, gMemPool);

    mScene = MakeShared<IDCSG_NS::Scene>(GetDGCSeqMgr(), *mGeoMgr, *mModelMgr,
                                         gMemPool);
}

DynamicLoading::~DynamicLoading() = default;

void DynamicLoading::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
}

void DynamicLoading::LoadShaders() {
    auto& shaderMgr = GetShaderMgr();

    shaderMgr.CreateShaderFromGLSL("computeDraw",
                                   SHADER_PATH_CSTR("BackGround.comp"),
                                   vk::ShaderStageFlagBits::eCompute);

    shaderMgr.CreateShaderFromGLSL("computeDraw-dgc-test",
                                   SHADER_PATH_CSTR("BackGround_DGCTest.comp"),
                                   vk::ShaderStageFlagBits::eCompute);

    shaderMgr.CreateShaderFromGLSL("vertex", SHADER_PATH_CSTR("Triangle.vert"),
                                   vk::ShaderStageFlagBits::eVertex, true);

    shaderMgr.CreateShaderFromGLSL("fragment",
                                   SHADER_PATH_CSTR("Triangle.frag"),
                                   vk::ShaderStageFlagBits::eFragment, true);

    shaderMgr.CreateShaderFromGLSL("Mesh shader fragment",
                                   SHADER_PATH_CSTR("MeshShader.frag"),
                                   vk::ShaderStageFlagBits::eFragment, true);

    Type_ShaderMacros taskMacros {};
    taskMacros.emplace("TASK_INVOCATION_COUNT",
                       std::to_string(TASK_SHADER_INVOCATION_COUNT).c_str());
    shaderMgr.CreateShaderFromGLSL(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        vk::ShaderStageFlagBits::eTaskEXT, false, taskMacros);

    Type_ShaderMacros meshMacros {};
    meshMacros.emplace("MESH_INVOCATION_COUNT",
                       std::to_string(MESH_SHADER_INVOCATION_COUNT).c_str());
    meshMacros.emplace(
        "MAX_VERTICES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES).c_str());
    meshMacros.emplace(
        "MAX_PRIMITIVES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES).c_str());
    shaderMgr.CreateShaderFromGLSL(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        vk::ShaderStageFlagBits::eMeshEXT, true, meshMacros);

    shaderMgr.CreateShaderFromGLSL("Quad vertex", SHADER_PATH_CSTR("Quad.vert"),
                                   vk::ShaderStageFlagBits::eVertex);

    shaderMgr.CreateShaderFromGLSL("Quad fragment",
                                   SHADER_PATH_CSTR("Quad.frag"),
                                   vk::ShaderStageFlagBits::eFragment);

    /**
     * shader objects
     */
    shaderMgr.CreateShaderObjectFromGLSL("computeDraw",
                                         SHADER_PATH_CSTR("BackGround.comp"),
                                         vk::ShaderStageFlagBits::eCompute);

    shaderMgr.CreateShaderObjectFromGLSL(
        "computeDraw-dgc-test", SHADER_PATH_CSTR("BackGround_DGCTest.comp"),
        vk::ShaderStageFlagBits::eCompute);

    shaderMgr.CreateShaderObjectFromGLSL(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        vk::ShaderStageFlagBits::eTaskEXT,
        vk::ShaderCreateFlagBitsEXT::eIndirectBindable, false, taskMacros);

    shaderMgr.CreateShaderObjectFromGLSL(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        vk::ShaderStageFlagBits::eMeshEXT,
        vk::ShaderCreateFlagBitsEXT::eIndirectBindable, true, meshMacros);

    shaderMgr.CreateShaderObjectFromGLSL(
        "Mesh shader fragment", SHADER_PATH_CSTR("MeshShader.frag"),
        vk::ShaderStageFlagBits::eFragment,
        vk::ShaderCreateFlagBitsEXT::eIndirectBindable, true);

    shaderMgr.CreateShaderObjectFromGLSL(
        "prepareDGC", SHADER_PATH_CSTR("PrepareGDCBuffer_test.comp"),
        vk::ShaderStageFlagBits::eCompute);
}

void DynamicLoading::PollEvents(SDL_Event* e, float deltaTime) {
    Application::PollEvents(e, deltaTime);

    switch (e->type) {
        case SDL_DROPFILE: {
            DBG_LOG_INFO("Dropped file: %s", e->drop.file);
            ::std::filesystem::path path {e->drop.file};

            AddNewNode(path.string().c_str());

            auto& node = mScene->GetNode(path.string().c_str());

            SDL_free(e->drop.file);
        } break;
        default: break;
    }

    GetUILayer().PollEvent(e);

    if (GetUILayer().WantCaptureKeyboard()) {
        mMainCamera->mCaptureKeyboard = false;
    } else {
        mMainCamera->mCaptureKeyboard = true;
    }

    mMainCamera->ProcessSDLEvent(e, deltaTime);
}

void DynamicLoading::Update_OnResize() {
    Application::Update_OnResize();

    auto& window = GetSDLWindow();
    auto& renderResMgr = GetRenderResMgr();

    vk::Extent2D extent = {static_cast<uint32_t>(window.GetWidth()),
                           static_cast<uint32_t>(window.GetHeight())};

    mMainCamera->SetAspect(extent.width / extent.height);

    renderResMgr.ResizeResources_ScreenSizeRelated(extent);

    auto resNames = renderResMgr.GetResourceNames_SrcreenSizeRelated();

    auto& backgroundPass = mRenderSequence.FindPass("DGC_Dispatch");
    backgroundPass.Update(resNames);

    auto& meshShaderPass = mRenderSequence.FindPass("DGC_DrawMeshShader");
    meshShaderPass.OnResize(extent);

    auto& drawQuadPass = mRenderSequence.FindPass("DrawQuad");
    drawQuadPass.OnResize(extent);

    mRenderSequence.GeneratePreRenderBarriers();
    mRenderSequence.ExecutePreRenderBarriers();
}

void DynamicLoading::UpdateScene() {
    Application::UpdateScene();

    mSceneData.cameraPos = {mMainCamera->mPosition.x, mMainCamera->mPosition.y,
                            mMainCamera->mPosition.z, 1.0f};
    mSceneData.view = mMainCamera->GetViewMatrix();
    mSceneData.proj = mMainCamera->GetProjectionMatrix();
    mSceneData.viewProj = mMainCamera->GetViewProjMatrix();

    UpdateSceneUBO();
}

void DynamicLoading::Prepare() {
    Application::Prepare();

    auto& window = GetSDLWindow();

    mRenderSequence.AddRenderPass("DGC_Dispatch");
    mRenderSequence.AddRenderPass("DGC_DrawMeshShader");
    mRenderSequence.AddRenderPass("DrawQuad");

    for (auto& frame : GetFrames()) {
        frame.PrepareBindlessDescPool({dynamic_cast<RenderPassBindingInfo_PSO*>(
            mRenderSequence.FindPass("DGC_DrawMeshShader").binding.get())});
    }

    CreateDrawImage();
    CreateDepthImage();

    LoadShaders();
    CreatePipelines();

    auto& renderResMgr = GetRenderResMgr();

    renderResMgr.CreateBuffer(
        "SceneUniformBuffer", sizeof(SceneData),
        vk::BufferUsageFlagBits::eUniformBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferSrc,
        Buffer::MemoryType::Staging);

    renderResMgr.CreateBuffer_ScreenSizeRelated(
        "RWBuffer", sizeof(float) * 4 * window.GetWidth() * window.GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eTransferSrc,
        Buffer::MemoryType::DeviceLocal, sizeof(float) * 4);

    {
        vk::DispatchIndirectCommand cmdBuffer {
            (uint32_t)::std::ceil(window.GetWidth() / 16.0),
            (uint32_t)::std::ceil(window.GetHeight() / 16.0), 1};

        mDispatchIndirectCmdBuffer =
            GetVulkanContext()
                .CreateIndirectCmdBuffer<vk::DispatchIndirectCommand>(
                    "dispatch command buffer", 1);
        auto bufSize = mDispatchIndirectCmdBuffer->GetSize();

        auto staging = GetVulkanContext().CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, &cmdBuffer, bufSize);

        {
            auto cmd = GetVulkanContext().CreateCmdBufToBegin(
                GetVulkanContext().GetQueue(QueueType::Transfer));
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(bufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mDispatchIndirectCmdBuffer->GetHandle(),
                            cmdBufCopy);
        }
    }

    // model0 = "29d64ac4-8943-4319-872f-31d275371cf6.fbx.cisdi";

    prepare_compute_sequence();
    prepare_dgc_draw_command();
    prepare_draw_mesh_task();

    PrepareUIContext();

    // RecordPasses(mRenderSequence);
}

void DynamicLoading::prepare_dgc_draw_command() {
    const uint32_t sequenceCount = 1;
    const uint32_t maxShaderCount = 1;
    const uint32_t maxDrawCount = 1;

    auto& dgcMgr = GetDGCSeqMgr();

    auto& sequence = dgcMgr.CreateSequence<PrepareDGCDrawCommandSequenceTemp>(
        {sequenceCount,
         maxDrawCount,
         maxShaderCount,
         {{"prepareDGC", vk::ShaderStageFlagBits::eCompute}}});

    // buffers
    {
        auto data = dgcMgr.CreateDataBuffer<PrepareDGCDrawCommandSequenceTemp>(
            "dgc_dispatch_for_draw");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].command = vk::DispatchIndirectCommand {1, 1, 1};
        }
    }
}

void DynamicLoading::prepare_compute_sequence() {
    const uint32_t sequenceCount = 1;
    const uint32_t maxPipelineCount = 1;
    const uint32_t maxDrawCount = 1;

    auto& dgcMgr = GetDGCSeqMgr();

    auto& sequence = dgcMgr.CreateSequence<DispatchSequenceTemp>(
        {sequenceCount,
         maxDrawCount,
         maxPipelineCount,
         {{"computeDraw", vk::ShaderStageFlagBits::eCompute}}});

    // buffers
    {
        auto data =
            dgcMgr.CreateDataBuffer<DispatchSequenceTemp>("dgc_dispatch_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].pushConstant = _baseColorFactor;
            data.data[i]
                .command.setX((uint32_t)std::ceil(1600 / 16.0))
                .setY((uint32_t)std::ceil(900 / 16.0))
                .setZ(1);
        }
    }
}

void DynamicLoading::prepare_draw_mesh_task() {
    Type_ShaderMacros taskMacros {};
    taskMacros.emplace("TASK_INVOCATION_COUNT",
                       std::to_string(TASK_SHADER_INVOCATION_COUNT).c_str());

    Type_ShaderMacros meshMacros {};
    meshMacros.emplace("MESH_INVOCATION_COUNT",
                       std::to_string(MESH_SHADER_INVOCATION_COUNT).c_str());
    meshMacros.emplace(
        "MAX_VERTICES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES).c_str());
    meshMacros.emplace(
        "MAX_PRIMITIVES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES).c_str());

    const uint32_t maxShaderCount = 1;

    auto& dgcMgr = GetDGCSeqMgr();

    dgcMgr.CreateSequence<DrawSequenceTemp>(
        {DGC_MAX_SEQUENCE_COUNT,
         DGC_MAX_DRAW_COUNT,
         maxShaderCount,
         {{"Mesh shader task", vk::ShaderStageFlagBits::eTaskEXT, taskMacros},
          {"Mesh shader mesh", vk::ShaderStageFlagBits::eMeshEXT, meshMacros},
          {"Mesh shader fragment", vk::ShaderStageFlagBits::eFragment}},
         true});
}

void DynamicLoading::ResizeToFitAllSeqBufPool() {
    for (auto const& seq : GetDGCSeqMgr().GetAllSequences()) {
        seq.second->GetBufferPool()->ResizeToFitIDAllocation();
    }

    for (auto& node : mScene->GetAllNodes()) {
        node.second->UploadSeqBuf();
    }
}

float DynamicLoading::AddNewNode(const char* modelPath) {
    float loadTime;

    Type_STLString path {modelPath};

    using Type_Promise = ::std::promise<
        IDVC_NS::UniquePtr<IDCSG_NS::NodeProxy<DrawSequenceTemp>>>;

    ::std::thread launch(
        [this, path](Type_Promise& p) {
            auto pBufferPool =
                GetDGCSeqMgr().GetSequence<DrawSequenceTemp>().GetBufferPool();

            auto node = IDCSG_NS::Node {gMemPool, path.c_str(), mScene.get()};
            auto nodeProxy = MakeUnique<IDCSG_NS::NodeProxy<DrawSequenceTemp>>(
                ::std::move(node), pBufferPool);

            auto const& modelData = nodeProxy->SetModel(path.c_str());

            p.set_value(::std::move(nodeProxy));
        },
        ::std::ref(mPromise));

    ::std::thread wait([this]() {
        auto& node = mScene->AddNodeProxy(mPromise.get_future().get());
        auto const& modelData = node.GetModel();
        AdjustCameraPosition(*mMainCamera, modelData.boundingBox);
        mPromise = Type_Promise {};
    });

    launch.detach();
    wait.detach();

    // GameTimer timer;
    // INTELLI_DS_MEASURE_DURATION_MS_START(timer);

    // auto pBufferPool =
    //     GetDGCSeqMgr().GetSequence<DrawSequenceTemp>().GetBufferPool();
    // auto& node = mScene->AddNodeProxy<DrawSequenceTemp>(modelPath, pBufferPool);
    //
    // auto const& modelData = node.SetModel(modelPath);
    //
    // // GetVulkanContext().GetDevice()->waitIdle();
    //
    // AdjustCameraPosition(*mMainCamera, modelData.boundingBox);
    //
    // INTELLI_DS_MEASURE_DURATION_MS_END_STORE(timer, loadTime);

    return 0.0f;
}

void DynamicLoading::RemoveNode(const char* nodeName) {
    // GetVulkanContext().GetDevice()->waitIdle();
    Type_STLString name {nodeName};
    ::std::thread t([this, name]() { mScene->RemoveNode(name.c_str()); });
    t.detach();
}

void DynamicLoading::BeginFrame(IDVC_NS::RenderFrame& frame) {
    Application::BeginFrame(frame);
    GetUILayer().BeginFrame(frame);
}

void DynamicLoading::RenderFrame(IDVC_NS::RenderFrame& frame) {
    auto& vkCtx = GetVulkanContext();
    auto& timelineSem = vkCtx.GetTimelineSemphore();
    auto& cmdMgr = GetCmdMgr();

    const uint64_t graphicsFinished = timelineSem.GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    ResizeToFitAllSeqBufPool();

    frame.ClearGPUGeoDataRefs();
    frame.mCmdStagings.clear();

    mScene->ClearInFrustumNodes();
    mScene->CullNode(mMainCamera->GetFrustum());

    for (auto& node : mScene->GetInFrustumNodes()) {
        if (true /*bCullPass*/) {
            frame.CullRegister(node->GetGPUGeoDataRef());

            for (auto const& n : node->GetCopyInfos()) {
                frame.mCmdStagings.push_back(n[mFrameNum % 3].srcName);
            }
        }
    }

    RecordPasses(mRenderSequence, frame);

    // Compute Draw
    {
        auto cmd = frame.GetGraphicsCmdBuf();

        frame.GetQueryPool().ResetPool(cmd.GetHandle(), 6);

        frame.GetQueryPool().BeginRange(cmd.GetHandle(), "dispatch");

        mRenderSequence.RecordPass("DGC_Dispatch", cmd.GetHandle());

        frame.GetQueryPool().EndRange(cmd.GetHandle(), "dispatch");

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             frame.GetPresentFinishedSemaphore().GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eBottomOfPipe, timelineSem.GetHandle(),
             graphicsFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllCommands, mCmpSem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eAllCommands, timelineSem.GetHandle(),
             computeFinished}};

        cmdMgr.Submit(cmd.GetHandle(),
                      vkCtx.GetQueue(QueueType::Graphics).GetHandle(), waits,
                      signals, frame.GetFencePool().RequestFence());
    }

    // Graphics Draw
    {
        auto cmd = frame.GetGraphicsCmdBuf();

        // mRenderSequence.RecordPass("DGC_PrepareDraw", cmd.GetHandle());

        frame.GetQueryPool().BeginRange(cmd.GetHandle(), "copy");

        mRenderSequence.RecordPass("dgc_seq_data_buf_copy", cmd.GetHandle());

        frame.GetQueryPool().EndRange(cmd.GetHandle(), "copy");

        frame.GetQueryPool().BeginRange(cmd.GetHandle(), "draw");

        mRenderSequence.RecordPass("DGC_DrawMeshShader", cmd.GetHandle());

        frame.GetQueryPool().EndRange(cmd.GetHandle(), "draw");

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eAllCommands, mCmpSem.GetHandle(), 0},
            {vk::PipelineStageFlagBits2::eComputeShader,
             timelineSem.GetHandle(), computeFinished}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics, timelineSem.GetHandle(),
             allFinished},
            {vk::PipelineStageFlagBits2::eAllGraphics,
             frame.GetRenderFinishedSemaphore().GetHandle()}};

        cmdMgr.Submit(cmd.GetHandle(),
                      vkCtx.GetQueue(QueueType::Graphics).GetHandle(), waits,
                      signals, frame.GetFencePool().RequestFence());
    }
}

void DynamicLoading::EndFrame(IDVC_NS::RenderFrame& frame) {
    Application::EndFrame(frame);

    frame.GetQueryPool().GetResult();
}

void DynamicLoading::RenderToSwapchainBindings(vk::CommandBuffer cmd) {
    mRenderSequence.GetRenderToSwapchainPass().RecordCmd(cmd);
}

void DynamicLoading::CreateDrawImage() {
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

    auto& vkCtx = GetVulkanContext();
    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueType::Graphics));
        Utils::TransitionImageLayout(cmd.mHandle, ref.GetTexHandle(),
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

void DynamicLoading::CreateDepthImage() {
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
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueType::Graphics));
        Utils::TransitionImageLayout(
            cmd.mHandle, ref.GetTexHandle(), vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }
}

void DynamicLoading::CreateBackgroundComputePipeline() {
    auto& shaderMgr = GetShaderMgr();

    auto compute =
        shaderMgr.GetShader("computeDraw", vk::ShaderStageFlagBits::eCompute);
    auto program = shaderMgr.CreateProgram("background", compute);

    vk::PipelineCreateFlags2CreateInfo pipelineFlags {};
    pipelineFlags.setFlags(vk::PipelineCreateFlagBits2::eIndirectBindableEXT);

    auto builder = GetPipelineMgr().GetBuilder_Compute();

    auto backgroundComputePipeline =
        builder.SetShaderProgram(program)
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background", {}, &pipelineFlags);

    compute = shaderMgr.GetShader("computeDraw-dgc-test",
                                  vk::ShaderStageFlagBits::eCompute);
    program = shaderMgr.CreateProgram("background-dgc-test", compute);

    auto builder2 = GetPipelineMgr().GetBuilder_Compute();

    auto backgroundComputePipeline2 =
        builder2.SetShaderProgram(program)
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background-dgctest", {}, &pipelineFlags);

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void DynamicLoading::CreateMeshShaderPipeline() {
    auto& renderResMgr = GetRenderResMgr();
    auto& shaderMgr = GetShaderMgr();

    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT).c_str());

    auto task = shaderMgr.GetShader("Mesh shader task",
                                    vk::ShaderStageFlagBits::eTaskEXT, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT).c_str());
    macros.emplace(
        "MAX_VERTICES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES).c_str());
    macros.emplace(
        "MAX_PRIMITIVES",
        std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES).c_str());

    auto mesh = shaderMgr.GetShader("Mesh shader mesh",
                                    vk::ShaderStageFlagBits::eMeshEXT, macros);

    auto frag = shaderMgr.GetShader("Mesh shader fragment",
                                    vk::ShaderStageFlagBits::eFragment);

    auto program = shaderMgr.CreateProgram("meshlet draw", task, mesh, frag);

    vk::PipelineCreateFlags2CreateInfo pipelineFlags {};
    pipelineFlags.setFlags(vk::PipelineCreateFlagBits2::eIndirectBindableEXT);

    auto builder = GetPipelineMgr().GetBuilder_Graphics();
    builder.SetShaderProgram(program)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::True)
        .SetDepth(vk::True, vk::True, vk::CompareOp::eGreaterOrEqual)
        .SetColorAttachmentFormat(renderResMgr["DrawImage"].GetTexFormat())
        .SetDepthStencilFormat(renderResMgr["DepthImage"].GetTexFormat())
        .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
        .Build("MeshShaderDraw", {}, &pipelineFlags);

    DBG_LOG_INFO("Vulkan MeshShader Graphics Pipeline Created");
}

void DynamicLoading::CreateDrawQuadPipeline() {
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

void DynamicLoading::UpdateSceneUBO() {
    auto& renderResMgr = GetRenderResMgr();
    auto data = renderResMgr["SceneUniformBuffer"].GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

namespace {

void DisplayNode(const IntelliDesign_NS::ModelData::CISDI_Node& node,
                 uint32_t idx,
                 const IntelliDesign_NS::ModelData::CISDI_3DModel& model) {
    if (ImGui::TreeNode(
            (node.name + "##" + std::to_string(idx).c_str()).c_str())) {
        if (node.meshIdx != -1) {
            const auto& mesh = model.meshes[node.meshIdx];

            // Display node
            ImGui::Text("vertex count: %d", mesh.header.vertexCount);
            ImGui::Text("meshlet count: %d", mesh.header.meshletCount);
            ImGui::Text("meshlet triangle count: %d",
                        mesh.header.meshletTriangleCount);
        }

        if (node.materialIdx != -1) {
            ImGui::Text("material: %s",
                        model.materials[node.materialIdx].name.c_str());
        }

        if (node.userPropertyCount > 0) {
            if (ImGui::TreeNode("User Properties:")) {
                for (const auto& [k, v] : node.userProperties) {
                    ImGui::Text(
                        "%s: %s", k.c_str(),
                        std::visit(
                            [](auto&& val) -> std::string {
                                using T = std::decay_t<decltype(val)>;
                                if constexpr (std::is_same_v<T,
                                                             Type_STLString>) {
                                    return {val.c_str()};
                                } else {
                                    return std::to_string(val);
                                }
                            },
                            v)
                            .c_str());
                }
                ImGui::TreePop();
            }
        }

        for (const auto& childIdx : node.childrenIdx) {
            DisplayNode(model.nodes[childIdx], childIdx, model);
        }
        ImGui::TreePop();
    }
}

}  // namespace

void DynamicLoading::PrepareUIContext() {
    auto& renderResMgr = GetRenderResMgr();

    GetUILayer()
        .AddContext([]() {
            ImGui::Begin("Guide");
            ImGui::Text("按 WASD 移动相机位置，按住鼠标右键控制相机朝向。");
            ImGui::End();
        })
        .AddFrameRelatedContext([this](IDVC_NS::RenderFrame& frame) {
            static float frameRate {};
            static float accumulatedTime {};

            static float dispatchTime {};
            static float copyTime {};
            static float drawTime {};

            accumulatedTime += 1000.0f / ImGui::GetIO().Framerate;

            if (accumulatedTime > 500.0f) {
                frameRate = ImGui::GetIO().Framerate;
                dispatchTime = frame.GetQueryPool().ElapsedTime("dispatch");
                copyTime = frame.GetQueryPool().ElapsedTime("copy");
                drawTime = frame.GetQueryPool().ElapsedTime("draw");
                accumulatedTime = 0.0f;
            }

            uint32_t totalNodeCount = mScene->GetAllNodes().size();
            uint32_t inFrustumNodeCount = mScene->GetInFrustumNodes().size();

            ImGui::Begin("渲染信息");
            {
                ImGui::Text("单帧耗时 %.3f ms/frame (%.1f FPS)",
                            1000.0f / frameRate, frameRate);

                ImGui::Text("\tDispatch pass 耗时 %.3f ms/frame.",
                            dispatchTime);
                ImGui::Text("\tCopy pass 耗时 %.3f ms/frame.", copyTime);
                ImGui::Text("\tDraw Model pass 耗时 %.3f ms/frame.", drawTime);

                static float loadTime {};

                if (ImGui::Button("浏览文件列表")) {
                    IGFD::FileDialogConfig config;
                    config.path = "../../../Models/股份数字化";
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseFileDlgKey", "选择文件", ".cisdi,.fbx", config);
                }

                // display
                if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {  // action if OK
                        std::string filePathName =
                            ImGuiFileDialog::Instance()->GetFilePathName();

                        loadTime = AddNewNode(filePathName.c_str());
                    }

                    // close
                    ImGuiFileDialog::Instance()->Close();
                }

                ImGui::Text("加载耗时: %.3f s", loadTime / 1000.0f);

                ImGui::Text("Frustum culling models: %d / %d.",
                            inFrustumNodeCount, totalNodeCount);
            }
            ImGui::End();
        })
        .AddContext([&]() {
            if (ImGui::Begin("场景信息")) {
                ImGui::Text("相机位置 [%.3f, %.3f, %.3f]",
                            mMainCamera->mPosition.x, mMainCamera->mPosition.y,
                            mMainCamera->mPosition.z);

                IDCMCore_NS::Float3 lightPos {IDCMCore_NS::Vector3Normalize(
                    {mSceneData.sunLightPos.x, mSceneData.sunLightPos.y,
                     mSceneData.sunLightPos.z})};

                auto theta = acos(lightPos.y);
                auto phi = atan2(lightPos.x, lightPos.z);

                ImGui::SetNextItemWidth(100);
                ImGui::SliderFloat("太阳光方向 θ", &theta, 0, MATH_PI_FLOAT);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::SliderFloat("太阳光方向 φ", &phi, 0, MATH_2PI_FLOAT);

                mSceneData.sunLightPos = {sin(theta) * sin(phi), cos(theta),
                                          cos(phi) * sin(theta), 1.0f};
            }
            ImGui::End();
        })
        .AddContext([&]() {
            if (ImGui::Begin("Model stats:"))
                for (auto const& [n, pNode] : mScene->GetAllNodes()) {
                    ::std::filesystem::path name {n.c_str()};
                    name = name.stem().stem();
                    auto const& model = pNode->GetModel();

                    bool nodeOpen = ImGui::TreeNode(name.string().c_str());

                    bool deleted = false;

                    ImGui::SameLine();

                    Type_STLString buttonLabel =
                        Type_STLString {"删除##"} + name.string().c_str();
                    if (ImGui::Button(buttonLabel.c_str())) {
                        RemoveNode(n.c_str());
                        deleted = true;
                    }

                    if (nodeOpen) {
                        auto const& geo =
                            *mGeoMgr->GetGPUGeometryData(name.string().c_str());

                        auto modelStats = geo.GetStats();

                        if (ImGui::TreeNode("Stats")) {

                            ImGui::Text("Mesh Count: %d.",
                                        modelStats.mMeshCount);
                            ImGui::Text("Meshlet Count: %d.",
                                        modelStats.mMeshletCount);
                            ImGui::Text("Meshlet triangle Count: %d.",
                                        modelStats.mMeshletTriangleCount);
                            ImGui::Text("Vertex Count: %d.",
                                        modelStats.mVertexCount);
                            ImGui::TreePop();
                        }

                        if (ImGui::TreeNode("Hierarchy")) {
                            auto const& node = model.nodes[0];
                            DisplayNode(node, 0, model);
                            ImGui::TreePop();
                        }
                        if (ImGui::TreeNode("Materials")) {
                            for (auto const& material : model.materials) {
                                if (ImGui::TreeNode(material.name.c_str())) {
                                    ImGui::Text("Ambient: (%.3f, %.3f, %.3f)",
                                                material.data.ambient.x,
                                                material.data.ambient.y,
                                                material.data.ambient.z);
                                    ImGui::Text("AmbientFactor: %.3f",
                                                material.data.ambient.w);
                                    ImGui::Text("Diffuse: (%.3f, %.3f, %.3f)",
                                                material.data.diffuse.x,
                                                material.data.diffuse.y,
                                                material.data.diffuse.z);
                                    ImGui::Text("DiffuseFactor: %.3f",
                                                material.data.diffuse.w);
                                    ImGui::Text("Specular: (%.3f, %.3f, %.3f)",
                                                material.data.specular.x,
                                                material.data.specular.y,
                                                material.data.specular.z);
                                    ImGui::Text("SpecularFactor: %.3f",
                                                material.data.specular.w);
                                    ImGui::Text("Emissive: (%.3f, %.3f, %.3f)",
                                                material.data.emissive.x,
                                                material.data.emissive.y,
                                                material.data.emissive.z);
                                    ImGui::Text("EmissiveFactor: %.3f",
                                                material.data.emissive.w);
                                    ImGui::Text(
                                        "Reflection: (%.3f, %.3f, %.3f)",
                                        material.data.reflection.x,
                                        material.data.reflection.y,
                                        material.data.reflection.z);
                                    ImGui::Text("ReflectionFactor: %.3f",
                                                material.data.reflection.w);
                                    ImGui::Text(
                                        "Transparency: (%.3f, %.3f, %.3f)",
                                        material.data.transparency.x,
                                        material.data.transparency.y,
                                        material.data.transparency.z);
                                    ImGui::Text("TransparencyFactor: %.3f",
                                                material.data.transparency.w);
                                    ImGui::Text("Shininess: %.3f",
                                                material.data.shininess);

                                    ImGui::TreePop();
                                }
                            }
                            ImGui::TreePop();
                        }
                        ImGui::TreePop();
                    }
                    if (deleted) {
                        break;
                    }
                }

            ImGui::End();
        });
}

void DynamicLoading::RecordPasses(RenderSequence& sequence,
                                  IDVC_NS::RenderFrame& frame) {
    sequence.Clear();

    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    auto& dgcMgr = GetDGCSeqMgr();

    RenderSequenceConfig cfg {};

    cfg.AddRenderPass(
           "DGC_Dispatch",
           dgcMgr.GetSequence<DispatchSequenceTemp>().GetPipelineLayout())
        .SetBinding("image", "DrawImage")
        .SetBinding("StorageBuffer", "RWBuffer")
        .SetDGCSeqBufs({"dgc_dispatch_test"});

    auto& drawMeshShaderSeq = dgcMgr.GetSequence<DrawSequenceTemp>();
    auto names =
        drawMeshShaderSeq.GetBufferPool()
            ->VisitPoolResources<Type_STLVector<const char*>>(
                [](DGCSeqBase::SequenceDataBufferPool::Type_PoolResources const&
                       resources) {
                    Type_STLVector<const char*> names {};
                    for (auto const& res : resources) {
                        names.emplace_back(res->GetName().c_str());
                    }
                    return names;
                });

    Type_STLVector<CopyPassConfig::CopyInfo> copyInfos {};
    uint32_t idx {0};

    if (!mScene->GetInFrustumNodes().empty())
        for (auto const& node : mScene->GetInFrustumNodes()) {
            for (auto const& info : node->GetCopyInfos()) {
                CopyPassConfig::CopyInfo copyInfo {frame.mCmdStagings[idx++],
                                                   info[0].dstName,
                                                   info[0].info, false};
                copyInfos.push_back(copyInfo);
            }
        }

    cfg.AddCopyPass("dgc_seq_data_buf_copy")
        .SetClearBuffer(names)
        .SetBinding(copyInfos);

    // cfg.AddRenderPass("DGC_PrepareDraw",
    //                   dgcMgr.GetSequence<PrepareDGCDrawCommandSequenceTemp>()
    //                       .GetPipelineLayout())
    //     .SetBinding("UBO", "prepareDGC_UBO")
    //     .SetBinding("StorageBuffer", "dgc_draw_test_0")
    //     .SetDGCSeqBufs({"dgc_dispatch_for_draw"});

    cfg.AddRenderPass("DGC_DrawMeshShader",
                      drawMeshShaderSeq.GetPipelineLayout())
        .SetBinding("UBO", "SceneUniformBuffer")
        .SetBinding("outFragColor", "DrawImage")
        .SetBinding("_Depth_", "DepthImage")
        .SetDGCPipelineInfo(DGCPipelineInfo {
            .colorBlendInfo = {0,
                               {vk::True},
                               {{vk::BlendFactor::eOneMinusSrcAlpha,
                                 vk::BlendFactor::eSrcAlpha, vk::BlendOp::eAdd,
                                 vk::BlendFactor::eOne, vk::BlendFactor::eZero,
                                 vk::BlendOp::eAdd}}},
            .viewport = {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f},
            .scissor = {{0, 0}, {width, height}}})
        .SetRenderArea({{0, 0}, {width, height}})
        .SetDGCSeqBufs(names);

    cfg.AddRenderPass("DrawQuad", "QuadDraw")
        .SetBinding("tex", "DrawImage")
        .SetBinding("outFragColor", "_Swapchain_");

    cfg.Compile(sequence);
}
