#include "DGCTest.h"

#include <random>

#include "Core/System/GameTimer.h"

using namespace IDVC_NS;

namespace {
::std::pmr::memory_resource* gMemPool = ::std::pmr::get_default_resource();

void AdjustCameraPosition(
    IDC_NS::Camera& camera,
    IntelliDesign_NS::ModelData::AABoundingBox const& bb) {
    auto center = bb.Center.GetSIMD();
    center = IDCMCore_NS::VectorMultiply(center, {0.01f, 0.01f, 0.01f});
    auto extent = bb.Extents.GetSIMD();
    extent = IDCMCore_NS::VectorMultiply(extent, {0.01f, 0.01f, 0.01f});
    camera.AdjustPosition(center, extent);
}

}  // namespace

DGCTest::DGCTest(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescSetPool(GetVulkanContext())),
      mRenderSequence(GetVulkanContext(), GetRenderResMgr(), GetPipelineMgr(),
                      mDescSetPool),
      mCopySem(GetVulkanContext()),
      mCmpSem(GetVulkanContext()) {
    mMainCamera = MakeUnique<IDC_NS::Camera>(IDC_NS::PersperctiveInfo {
        1000.0f, 0.01f, IDCMCore_NS::ConvertToRadians(45.0f),
        (float)spec.width / spec.height});

    mScene = MakeShared<IDCSG_NS::Scene>(gMemPool);

    mModelMgr =
        MakeUnique<IntelliDesign_NS::ModelData::ModelDataManager>(gMemPool);

    mGeoMgr = MakeUnique<GPUGeometryDataManager>(GetVulkanContext(), gMemPool);

    mDGCSequenceMgr =
        MakeUnique<DGCSeqManager>(GetVulkanContext(), GetPipelineMgr(),
                                  GetShaderMgr(), GetRenderResMgr());
}

DGCTest::~DGCTest() = default;

void DGCTest::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
}

void DGCTest::LoadShaders() {
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
}

void DGCTest::PollEvents(SDL_Event* e, float deltaTime) {
    Application::PollEvents(e, deltaTime);

    switch (e->type) {
        case SDL_DROPFILE: {
            DBG_LOG_INFO("Dropped file: %s", e->drop.file);
            ::std::filesystem::path path {e->drop.file};

            GetVulkanContext().GetDevice()->waitIdle();

            auto& node = mScene->AddNode(path.stem().string().c_str());
            auto const& modelData =
                node.SetModel(path.string().c_str(), *mGeoMgr, *mModelMgr);

            AdjustCameraPosition(*mMainCamera, modelData.boundingBox);

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

void DGCTest::Update_OnResize() {
    Application::Update_OnResize();

    auto& window = GetSDLWindow();
    auto& renderResMgr = GetRenderResMgr();

    vk::Extent2D extent = {static_cast<uint32_t>(window.GetWidth()),
                           static_cast<uint32_t>(window.GetHeight())};

    mMainCamera->SetAspect(extent.width / extent.height);

    renderResMgr.ResizeResources_ScreenSizeRelated(extent);

    auto resNames = renderResMgr.GetResourceNames_SrcreenSizeRelated();

    auto& backgroundPass = mRenderSequence.FindPass("DrawBackground");
    backgroundPass.Update(resNames);

    auto& meshShaderPass = mRenderSequence.FindPass("DrawMeshShader");
    meshShaderPass.OnResize(extent);

    auto& drawQuadPass = mRenderSequence.FindPass("DrawQuad");
    drawQuadPass.OnResize(extent);

    mRenderSequence.GeneratePreRenderBarriers();
    mRenderSequence.ExecutePreRenderBarriers();
}

void DGCTest::UpdateScene() {
    Application::UpdateScene();

    mSceneData.cameraPos = {mMainCamera->mPosition.x, mMainCamera->mPosition.y,
                            mMainCamera->mPosition.z, 1.0f};
    mSceneData.view = mMainCamera->GetViewMatrix();
    mSceneData.proj = mMainCamera->GetProjectionMatrix();
    mSceneData.viewProj = mMainCamera->GetViewProjMatrix();

    UpdateSceneUBO();
}

void DGCTest::Prepare() {
    Application::Prepare();

    auto& window = GetSDLWindow();

    mRenderSequence.AddRenderPass("DrawBackground");
    mRenderSequence.AddRenderPass("DrawMeshShader");
    mRenderSequence.AddRenderPass("DrawQuad");

    for (auto& frame : GetFrames()) {
        frame.PrepareBindlessDescPool({dynamic_cast<RenderPassBindingInfo_PSO*>(
            mRenderSequence.FindPass("DrawMeshShader").binding.get())});
    }
    CreateRandomTexture();

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

    prepare_compute_sequence();
    prepare_compute_sequence_pipeline();
    prepare_compute_sequence_shader();

    const char* modelPath = "5d9b133d-bc33-42a1-86fe-3dc6996d5b46.fbx.cisdi";

    auto& node = mScene->AddNode("cisdi");
    auto const& modelData =
        node.SetModel(MODEL_PATH_CSTR(modelPath), *mGeoMgr, *mModelMgr);

    AdjustCameraPosition(*mMainCamera, modelData.boundingBox);

    prepare_draw_mesh_task();
    prepare_draw_mesh_task_pipeline();
    prepare_draw_mesh_task_shader();

    PrepareUIContext();

    // RecordPasses(mRenderSequence);
}

void DGCTest::prepare_compute_sequence() {
    const uint32_t sequenceCount = 2;
    const uint32_t maxPipelineCount = 1;
    const uint32_t maxDrawCount = 1;

    auto& sequence = mDGCSequenceMgr->CreateSequence<DispatchSequenceTemp>(
        {sequenceCount,
         maxDrawCount,
         maxPipelineCount,
         {{"computeDraw", vk::ShaderStageFlagBits::eCompute}}});

    // buffers
    {
        auto data = mDGCSequenceMgr->CreateDataBuffer<DispatchSequenceTemp>(
            "dgc_dispatch_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].pushConstant = _baseColorFactor;
            data.data[i]
                .command.setX((uint32_t)std::ceil(1600 / 16.0))
                .setY((uint32_t)std::ceil(900 / 16.0))
                .setZ(1);
        }
    }
}

void DGCTest::prepare_draw_mesh_task() {
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

    const uint32_t sequenceCount = 3;
    const uint32_t maxShaderCount = 3;
    const uint32_t maxDrawCount = 2000;

    auto& sequence = mDGCSequenceMgr->CreateSequence<DrawSequenceTemp>(
        {sequenceCount,
         maxDrawCount,
         maxShaderCount,
         {{"Mesh shader task", vk::ShaderStageFlagBits::eTaskEXT, taskMacros},
          {"Mesh shader mesh", vk::ShaderStageFlagBits::eMeshEXT, meshMacros},
          {"Mesh shader fragment", vk::ShaderStageFlagBits::eFragment}}, true});

    // buffers
    {
        auto const& node = mScene->GetNode("cisdi");
        auto const& modelName = node.GetModel().name;
        auto& geo = mGeoMgr->GetGPUGeometryData(modelName.c_str());
        auto pushConstant = *geo.GetMeshletPushContantsPtr();

        ::std::array<MeshletPushConstants, 3> meshletConstants {};

        meshletConstants[0] = pushConstant;
        meshletConstants[0].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f);

        meshletConstants[1] = pushConstant;
        meshletConstants[1].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f)
            * IDCMCore_NS::MatrixTranslation(10.0f, 10.0f, 10.0f);

        meshletConstants[2] = pushConstant;
        meshletConstants[2].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f)
            * IDCMCore_NS::MatrixTranslation(20.0f, 20.0f, 20.0f);

        auto command = geo.GetDrawIndirectCmdBufInfo();

        auto data = mDGCSequenceMgr->CreateDataBuffer<DrawSequenceTemp>(
                "dgc_draw_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].pushConstant = meshletConstants[i];
            data.data[i].command = command;
        }
    }
}

void DGCTest::prepare_compute_sequence_pipeline() {
    const uint32_t sequenceCount = 2;
    const uint32_t maxPipelineCount = 2;
    const uint32_t maxDrawCount = 1;

    auto& sequence =
        mDGCSequenceMgr->CreateSequence<DispatchSequence_PipelineTemp>(
            {sequenceCount, maxDrawCount, maxPipelineCount, "Background"},
            {"Background-dgctest"});

    // buffers
    {
        auto data =
            mDGCSequenceMgr->CreateDataBuffer<DispatchSequence_PipelineTemp>(
                "dgc_pipe_dispatch_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].index = i;
            data.data[i].pushConstant = _baseColorFactor;
            data.data[i]
                .command.setX((uint32_t)std::ceil(1600 / 16.0))
                .setY((uint32_t)std::ceil(900 / 16.0))
                .setZ(1);
        }
    }
}

void DGCTest::prepare_draw_mesh_task_pipeline() {
    const uint32_t sequenceCount = 2;
    const uint32_t maxPipelineCount = 2;
    const uint32_t maxDrawCount = 2000;

    auto& sequence = mDGCSequenceMgr->CreateSequence<DrawSequence_PipelineTemp>(
        {sequenceCount, maxDrawCount, maxPipelineCount, "MeshShaderDraw",
         true});

    // buffers
    {
        auto const& node = mScene->GetNode("cisdi");
        auto const& modelName = node.GetModel().name;
        auto& geo = mGeoMgr->GetGPUGeometryData(modelName.c_str());
        auto pushConstant = *geo.GetMeshletPushContantsPtr();

        ::std::array<MeshletPushConstants, 2> meshletConstants {};

        meshletConstants[0] = pushConstant;
        meshletConstants[0].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f);

        meshletConstants[1] = pushConstant;
        meshletConstants[1].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f)
            * IDCMCore_NS::MatrixTranslation(10.0f, 10.0f, 10.0f);

        auto command = geo.GetDrawIndirectCmdBufInfo();

        auto data =
            mDGCSequenceMgr->CreateDataBuffer<DrawSequence_PipelineTemp>(
                "dgc_pipe_draw_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].index = i;
            data.data[i].pushConstant = meshletConstants[i];
            data.data[i].command = command;
        }
    }
}

void DGCTest::prepare_compute_sequence_shader() {
    const uint32_t sequenceCount = 2;
    const uint32_t maxShaderCount = 2;
    const uint32_t maxDrawCount = 1;

    auto& sequence =
        mDGCSequenceMgr->CreateSequence<DispatchSequence_ShaderTemp>(
            {sequenceCount,
             maxDrawCount,
             maxShaderCount,
             {{"computeDraw", vk::ShaderStageFlagBits::eCompute}}},
            {{"computeDraw-dgc-test", vk::ShaderStageFlagBits::eCompute}});

    // buffers
    {
        auto data =
            mDGCSequenceMgr->CreateDataBuffer<DispatchSequence_ShaderTemp>(
                "dgc_shader_dispatch_test");

        for (uint32_t i = 0; i < sequence.GetSequenceCount(); ++i) {
            data.data[i].index = {i};
            data.data[i].pushConstant = _baseColorFactor;
            data.data[i]
                .command.setX((uint32_t)std::ceil(1600 / 16.0))
                .setY((uint32_t)std::ceil(900 / 16.0))
                .setZ(1);
        }
    }
}

void DGCTest::prepare_draw_mesh_task_shader() {
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

    const uint32_t sequenceCount = 1;
    const uint32_t maxShaderCount = 3;
    const uint32_t maxDrawCount = 2000;

    auto& sequence = mDGCSequenceMgr->CreateSequence<DrawSequence_ShaderTemp>(
        {sequenceCount,
         maxDrawCount,
         maxShaderCount,
         {{"Mesh shader task", vk::ShaderStageFlagBits::eTaskEXT, taskMacros},
          {"Mesh shader mesh", vk::ShaderStageFlagBits::eMeshEXT, meshMacros},
          {"Mesh shader fragment", vk::ShaderStageFlagBits::eFragment}},
         true});

    // buffers
    {
        auto const& node = mScene->GetNode("cisdi");
        auto const& modelName = node.GetModel().name;
        auto& geo = mGeoMgr->GetGPUGeometryData(modelName.c_str());
        auto pushConstant = *geo.GetMeshletPushContantsPtr();

        ::std::array<MeshletPushConstants, 2> meshletConstants {};

        meshletConstants[0] = pushConstant;
        meshletConstants[0].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f);

        meshletConstants[1] = pushConstant;
        meshletConstants[1].mModelMatrix =
            IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f)
            * IDCMCore_NS::MatrixTranslation(10.0f, 10.0f, 10.0f);

        auto command = geo.GetDrawIndirectCmdBufInfo();

        auto data = mDGCSequenceMgr->CreateDataBuffer<DrawSequence_ShaderTemp>(
            "dgc_shader_draw_test");

        for (uint32_t i = 0; i < 1; ++i) {
            data.data[i].index = {0, 1, 2};
            data.data[i].pushConstant = meshletConstants[i % 2];
            data.data[i].command = command;
        }
    }
}

void DGCTest::BeginFrame(IDVC_NS::RenderFrame& frame) {
    Application::BeginFrame(frame);
    GetUILayer().BeginFrame();
}

void DGCTest::RenderFrame(IDVC_NS::RenderFrame& frame) {
    auto& vkCtx = GetVulkanContext();
    auto& timelineSem = vkCtx.GetTimelineSemphore();
    auto& cmdMgr = GetCmdMgr();

    const uint64_t graphicsFinished = timelineSem.GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    RecordPasses(mRenderSequence);

    // Compute Draw
    {
        auto cmd = frame.GetGraphicsCmdBuf();

        mRenderSequence.RecordPass("DGC_Dispatch", cmd.GetHandle());

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

        mRenderSequence.RecordPass("DGC_DrawMeshShader", cmd.GetHandle());

        auto cmdHandle = cmd.GetHandle();

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

void DGCTest::EndFrame(IDVC_NS::RenderFrame& frame) {
    Application::EndFrame(frame);
}

void DGCTest::RenderToSwapchainBindings(vk::CommandBuffer cmd) {
    mRenderSequence.GetRenderToSwapchainPass().RecordCmd(cmd);
}

void DGCTest::CreateDrawImage() {
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

void DGCTest::CreateDepthImage() {
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

void DGCTest::CreateRandomTexture() {
    constexpr auto extent = vk::Extent3D {4, 4, 1};
    constexpr uint32_t randomImageCount = 16;

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
    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueType::Graphics));

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

            vk::CopyBufferToImageInfo2 info {};
            info.setSrcBuffer(GetRenderResMgr()["staging"].GetBufferHandle())
                .setDstImage(GetRenderResMgr()[name.c_str()].GetTexHandle())
                .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
                .setRegions(copyRegion);

            cmd->copyBufferToImage2(info);

            Utils::TransitionImageLayout(
                cmd.mHandle, renderResMgr[name.c_str()].GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }

    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        for (uint32_t j = 0; j < randomImageCount / 2; ++j) {
            auto name = baseName + ::std::to_string(j);
            auto texture = renderResMgr[name.c_str()].GetTexPtr();

            GetFrames()[i].GetBindlessDescPool().Add(texture);
        }
    }
}

void DGCTest::CreateBackgroundComputePipeline() {
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

void DGCTest::CreateMeshShaderPipeline() {
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

void DGCTest::CreateDrawQuadPipeline() {
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

void DGCTest::UpdateSceneUBO() {
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

void DGCTest::PrepareUIContext() {
    mImageName0 = "RandomImage";
    mImageName1 = "RandomImage";
    auto& renderResMgr = GetRenderResMgr();
    GetUILayer()
        .AddContext([]() {
            ImGui::Begin("Guide");
            ImGui::Text("按 WASD 移动相机位置，按住鼠标右键控制相机朝向。");
            ImGui::End();
        })
        .AddContext([this]() {
            ImGui::Begin("渲染信息");
            {
                ImGui::Text("单帧耗时 %.3f ms/frame (%.1f FPS)",
                            1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);

                static char buf[1024] {};
                static float loadTime {};

                ImGui::SetNextItemWidth(200);
                ImGui::InputText("模型文件名 -->", buf, 1024);

                ::std::string buffName = buf;

                ImGui::SameLine();
                if (ImGui::Button("加载")) {
                    if (buffName.empty()
                        || !::std::filesystem::exists(
                            MODEL_PATH_CSTR(buffName.c_str()))) {
                        MessageBoxW(nullptr, L"模型文件不存在", L"错误", MB_OK);
                    } else {
                        GameTimer timer;
                        INTELLI_DS_MEASURE_DURATION_MS_START(timer);

                        GetVulkanContext().GetDevice()->waitIdle();

                        auto& node = mScene->GetNode("cisdi");
                        auto const& modelData =
                            node.SetModel(MODEL_PATH_CSTR(buffName.c_str()),
                                          *mGeoMgr, *mModelMgr);

                        AdjustCameraPosition(*mMainCamera,
                                             modelData.boundingBox);

                        INTELLI_DS_MEASURE_DURATION_MS_END_STORE(timer,
                                                                 loadTime);
                    }
                }

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
                        std::string filePath =
                            ImGuiFileDialog::Instance()->GetCurrentPath();

                        GameTimer timer;
                        INTELLI_DS_MEASURE_DURATION_MS_START(timer);

                        GetVulkanContext().GetDevice()->waitIdle();

                        auto& node = mScene->GetNode("cisdi");
                        auto const& modelData = node.SetModel(
                            filePathName.c_str(), *mGeoMgr, *mModelMgr);

                        AdjustCameraPosition(*mMainCamera,
                                             modelData.boundingBox);

                        INTELLI_DS_MEASURE_DURATION_MS_END_STORE(timer,
                                                                 loadTime);
                    }

                    // close
                    ImGuiFileDialog::Instance()->Close();
                }

                ImGui::Text("加载耗时: %.3f s", loadTime / 1000.0f);
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
            auto& model = mScene->GetNode("cisdi").GetModel();

            if (ImGui::Begin(model.name.c_str())) {
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
                            ImGui::Text("Reflection: (%.3f, %.3f, %.3f)",
                                        material.data.reflection.x,
                                        material.data.reflection.y,
                                        material.data.reflection.z);
                            ImGui::Text("ReflectionFactor: %.3f",
                                        material.data.reflection.w);
                            ImGui::Text("Transparency: (%.3f, %.3f, %.3f)",
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
            }
            ImGui::End();
        });
}

void DGCTest::RecordPasses(RenderSequence& sequence) {
    sequence.Clear();

    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    RenderSequenceConfig cfg {};

    cfg.AddRenderPass("DGC_Dispatch",
                      &GetRenderResMgr()["dgc_dispatch_test"])
        .SetBinding("image", "DrawImage")
        .SetBinding("StorageBuffer", "RWBuffer");

    auto bindlessSet = GetCurFrame().GetBindlessDescPool().GetPoolResource();

    cfg.AddRenderPass("DGC_DrawMeshShader",
                      &GetRenderResMgr()["dgc_draw_test"])
        .SetBinding("UBO", "SceneUniformBuffer")
        // .SetBinding("sceneTexs", {bindlessSet.deviceAddr, bindlessSet.offset})
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
        .SetRenderArea({{0, 0}, {width, height}});

    // cfg.AddRenderPass("DGC_DrawMeshShader",
    //                   &GetRenderResMgr()["dgc_draw_test"])
    //     .SetBinding("UBO", "SceneUniformBuffer")
    //     // .SetBinding("sceneTexs", {bindlessSet.deviceAddr, bindlessSet.offset})
    //     .SetBinding("outFragColor", "DrawImage")
    //     .SetBinding("_Depth_", "DepthImage")
    //     .SetViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f})
    //     .SetScissor({{0, 0}, {width, height}})
    //     .SetRenderArea({{0, 0}, {width, height}});

    cfg.AddRenderPass("DrawQuad", "QuadDraw")
        .SetBinding("tex", "DrawImage")
        .SetBinding("outFragColor", "_Swapchain_");

    cfg.Compile(sequence);
}
