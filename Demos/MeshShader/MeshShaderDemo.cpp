#include "MeshShaderDemo.h"

#include <random>

#include <glm/gtc/packing.hpp>

#include "Core/System/GameTimer.h"

using namespace IDNS_VC;

MeshShaderDemo::MeshShaderDemo(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescSetPool(GetVulkanContext())),
      mRenderSequence(GetVulkanContext(), GetRenderResMgr(), GetPipelineMgr(),
                      mDescSetPool),
      mCopySem(GetVulkanContext()),
      mCmpSem(GetVulkanContext()) {}

MeshShaderDemo::~MeshShaderDemo() = default;

void MeshShaderDemo::CreatePipelines() {
    CreateBackgroundComputePipeline();
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

    switch (e->type) {
        case SDL_DROPFILE: {
            DBG_LOG_INFO("Dropped file: %s", e->drop.file);
            ::std::filesystem::path path{e->drop.file};

            GetVulkanContext().GetDevice()->waitIdle();

            auto pMemPool = ::std::pmr::get_default_resource();

            mFactoryModel = MakeShared<Geometry>(path.string().c_str(), pMemPool);
            mFactoryModel->GenerateMeshletBuffers(GetVulkanContext());

            SDL_free(e->drop.file);
        } break;
        default: break;
    }

    GetUILayer().PollEvent(e);

    if (GetUILayer().WantCaptureKeyboard()) {
        mMainCamera.mCaptureKeyboard = false;
    } else {
        mMainCamera.mCaptureKeyboard = true;
    }

    mMainCamera.ProcessSDLEvent(e, deltaTime);
}

void MeshShaderDemo::Update_OnResize() {
    Application::Update_OnResize();

    auto& window = GetSDLWindow();
    auto& renderResMgr = GetRenderResMgr();

    vk::Extent2D extent = {static_cast<uint32_t>(window.GetWidth()),
                           static_cast<uint32_t>(window.GetHeight())};

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

void MeshShaderDemo::UpdateScene() {
    Application::UpdateScene();

    auto& window = GetSDLWindow();

    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(window.GetWidth())
                             / static_cast<float>(window.GetHeight()),
                         1000.0f, 0.01f);

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

    mRenderSequence.AddRenderPass("DrawBackground", RenderQueueType::Compute);
    mRenderSequence.AddRenderPass("DrawMeshShader", RenderQueueType::Graphics);
    mRenderSequence.AddRenderPass("DrawQuad", RenderQueueType::Graphics);

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
        "RWBuffer", sizeof(glm::vec4) * window.GetWidth() * window.GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst,
        Buffer::MemoryType::DeviceLocal, sizeof(glm::vec4));

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

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // cisdi model data converter
    {
        IntelliDesign_NS::Core::Utils::Timer timer;

        const char* model = "5d9b133d-bc33-42a1-86fe-3dc6996d5b46.fbx.cisdi";

        auto pMemPool = ::std::pmr::get_default_resource();

        mFactoryModel = MakeShared<Geometry>(MODEL_PATH_CSTR(model), pMemPool);

        auto duration_LoadModel = timer.End();
        printf("Load Geometry: %s, Time consumed: %f s. \n", model,
               duration_LoadModel);

        mFactoryModel->GenerateMeshletBuffers(GetVulkanContext());
    }

    PrepareUIContext();

    // RecordPasses(mRenderSequence);
}

void MeshShaderDemo::BeginFrame(IDNS_VC::RenderFrame& frame) {
    Application::BeginFrame(frame);
    GetUILayer().BeginFrame();
}

void MeshShaderDemo::RenderFrame(IDNS_VC::RenderFrame& frame) {
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

        mRenderSequence.RecordPass("DrawBackground", cmd.GetHandle());

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

        mRenderSequence.RecordPass("DrawMeshShader", cmd.GetHandle());

        // mRenderSequence.RecordPass("Copytest", cmd.GetHandle());
        mRenderSequence.RecordPass("executortest", cmd.GetHandle());

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

    GetVulkanContext().GetDevice()->waitIdle();

    // {
    //     if(mFrameNum % 1000 ==0) {
    //         const char* model =
    //             "5d9b133d-bc33-42a1-86fe-3dc6996d5b46.fbx.cisdi";
    //
    //         auto pMemPool = ::std::pmr::get_default_resource();
    //
    //         mFactoryModel =
    //             MakeShared<Geometry>(MODEL_PATH_CSTR(model), pMemPool);
    //
    //         mFactoryModel->GenerateMeshletBuffers(GetVulkanContext());
    //     }
    // }
}

void MeshShaderDemo::EndFrame(IDNS_VC::RenderFrame& frame) {
    Application::EndFrame(frame);
}

void MeshShaderDemo::RenderToSwapchainBindings(vk::CommandBuffer cmd) {
    mRenderSequence.GetRenderToSwapchainPass().RecordCmd(cmd);
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

    auto& vkCtx = GetVulkanContext();
    {
        auto cmd =
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueType::Graphics));
        Utils::TransitionImageLayout(cmd.mHandle, ref.GetTexHandle(),
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eShaderReadOnlyOptimal);
    }
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
            vkCtx.CreateCmdBufToBegin(vkCtx.GetQueue(QueueType::Graphics));
        Utils::TransitionImageLayout(
            cmd.mHandle, ref.GetTexHandle(), vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }
}

void MeshShaderDemo::CreateRandomTexture() {
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
        .SetBlending(vk::True)
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

void MeshShaderDemo::UpdateSceneUBO() {
    auto& renderResMgr = GetRenderResMgr();
    auto data = renderResMgr["SceneUniformBuffer"].GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

void MeshShaderDemo::PrepareUIContext() {
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

                ImGui::InputText("模型文件名 -->", buf, 1024);

                ::std::string buffName = buf;

                ImGui::SameLine();
                if (ImGui::Button("加载")) {
                    if (buffName.empty()
                        || !::std::filesystem::exists(
                            MODEL_PATH_CSTR(buffName.c_str()))) {
                        MessageBoxW(nullptr, L"模型文件不存在", L"错误", MB_OK);
                    } else {
                        auto pMemPool = ::std::pmr::get_default_resource();

                        GameTimer timer;
                        INTELLI_DS_MEASURE_DURATION_MS_START(timer);

                        mFactoryModel = MakeShared<Geometry>(
                            MODEL_PATH_CSTR(buffName.c_str()), pMemPool);
                        mFactoryModel->GenerateMeshletBuffers(
                            GetVulkanContext());

                        INTELLI_DS_MEASURE_DURATION_MS_END_STORE(timer,
                                                                 loadTime);
                    }
                }

                ImGui::Text("加载耗时: %.3f s", loadTime / 1000.0f);
            }
            ImGui::End();
        })
        .AddContext([&]() {
            if (ImGui::Begin("SceneStats")) {
                ImGui::Text("Camera Position: (%.3f, %.3f, %.3f)",
                            mSceneData.cameraPos.x, mSceneData.cameraPos.y,
                            mSceneData.cameraPos.z);
                ImGui::SliderFloat3("Sun light position",
                                    (float*)&mSceneData.sunLightPos, -1.0f,
                                    1.0f);
                ImGui::ColorEdit4("ObjColor", (float*)&mSceneData.objColor);
                ImGui::SliderFloat2("MetallicRoughness",
                                    (float*)&mSceneData.metallicRoughness, 0.0f,
                                    1.0f);
                ImGui::InputInt("Texture Index", &mSceneData.texIndex);

                ImGui::InputText("##1", mImageName0.data(), 32);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    auto tex = renderResMgr[mImageName0.c_str()].GetTexPtr();
                    auto idx = GetCurFrame().GetBindlessDescPool().Add(tex);

                    DBG_LOG_INFO("Add Button pressed, add texture at idx %d",
                                 idx);
                }

                ImGui::InputText("##2", mImageName1.data(), 32);
                ImGui::SameLine();
                if (ImGui::Button("Delete")) {
                    auto tex = renderResMgr[mImageName1.c_str()].GetTexPtr();
                    auto idx = GetCurFrame().GetBindlessDescPool().Delete(tex);
                    DBG_LOG_INFO(
                        "Delete Button pressed, delete texture at idx %d", idx);
                }
            }
            ImGui::End();
        })
        .AddContext([&]() {
            auto const& model = mFactoryModel->GetCISDIModelData();

            auto displayNode =
                [&,
                 d = [&](auto&& self,
                         IntelliDesign_NS::ModelData::CISDI_Node const& node,
                         uint32_t idx) -> void {
                     if (ImGui::TreeNode(
                             (node.name + "##" + ::std::to_string(idx).c_str())
                                 .c_str())) {

                         if (node.meshIdx != -1) {
                             auto const& mesh = model.meshes[node.meshIdx];

                             // display node
                             ImGui::Text("vertex count: %d",
                                         mesh.header.vertexCount);
                             ImGui::Text("meshlet count: %d",
                                         mesh.header.meshletCount);

                             ImGui::Text("meshlet triangle count: %d",
                                         mesh.header.meshletTriangleCount);
                         }

                         if (node.materialIdx != -1) {
                             ImGui::Text("material: %s",
                                         model.materials[node.materialIdx]
                                             .name.c_str());
                         }

                         if (node.userPropertyCount > 0) {
                             if (ImGui::TreeNode("User Properties:")) {
                                 for (auto const& [k, v] :
                                      node.userProperties) {
                                     ImGui::Text(
                                         "%s: %s", k.c_str(),
                                         ::std::visit(
                                             [&](auto&& val) -> ::std::string {
                                                 using T = std::decay_t<
                                                     decltype(val)>;
                                                 if constexpr (
                                                     ::std::is_same_v<
                                                         T, Type_STLString>) {
                                                     return {val.c_str()};
                                                 } else {
                                                     return ::std::to_string(
                                                         val);
                                                 }
                                             },
                                             v)
                                             .c_str());
                                 }
                                 ImGui::TreePop();
                             }
                         }

                         for (auto const& childIdx : node.childrenIdx) {
                             self(self, model.nodes[childIdx], childIdx);
                         }
                         ImGui::TreePop();
                     }
                 }](IntelliDesign_NS::ModelData::CISDI_Node const& node,
                    uint32_t idx) {
                    d(d, node, idx);
                };

            if (ImGui::Begin(model.name.c_str())) {
                if (ImGui::TreeNode("Hierarchy")) {
                    auto const& node = model.nodes[0];
                    displayNode(node, 0);
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

void MeshShaderDemo::RecordPasses(RenderSequence& sequence) {
    sequence.Clear();

    auto& drawImage = GetRenderResMgr()["DrawImage"];
    uint32_t width = drawImage.GetTexWidth();
    uint32_t height = drawImage.GetTexHeight();

    RenderSequenceConfig cfg {};

    cfg.AddRenderPass("DrawBackground", "Background")
        .SetBinding("image", "DrawImage")
        .SetBinding("StorageBuffer", "RWBuffer")
        .SetBinding(mDispatchIndirectCmdBuffer.get())
        .SetExecuteInfo(RenderPassConfig::ExecuteType::Dispatch);

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    meshPushContants->mModelMatrix =
        glm::scale(glm::mat4 {1.0f}, glm::vec3 {.01f});
    // meshPushContants->mModelMatrix =
    //     glm::rotate(meshPushContants->mModelMatrix, glm::radians(90.0f),
    //                 glm::vec3(-1.0f, 0.0f, 0.0f));

    auto bindlessSet = GetCurFrame().GetBindlessDescPool().GetPoolResource();

    cfg.AddRenderPass("DrawMeshShader", "MeshShaderDraw")
        .SetBinding("PushConstants", meshPushContants)
        .SetBinding("PushConstantsFrag",
                    mFactoryModel->GetFragmentPushConstantsPtr())
        .SetBinding("UBO", "SceneUniformBuffer")
        // .SetBinding("sceneTexs", {bindlessSet.deviceAddr, bindlessSet.offset})
        .SetBinding("outFragColor", "DrawImage")
        .SetBinding("_Depth_", "DepthImage")
        .SetBinding(mFactoryModel->GetMeshTaskIndirectCmdBuffer())
        .SetViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f})
        .SetScissor({{0, 0}, {width, height}})
        .SetRenderArea({{0, 0}, {width, height}})
        .SetExecuteInfo(RenderPassConfig::ExecuteType::DrawMeshTask);

    // cfg.AddCopyPass("Copytest")
    //     .SetBinding(
    //         {"SceneUniformBuffer", "RWBuffer", vk::BufferCopy2 {0, 0, 16}});

    cfg.AddExecutor("executortest")
        .SetBinding({{"SceneUniformBuffer", vk::ImageLayout::eUndefined,
                      vk::AccessFlagBits2::eTransferRead,
                      vk::PipelineStageFlagBits2::eTransfer},
                     {"RWBuffer", vk::ImageLayout::eUndefined,
                      vk::AccessFlagBits2::eTransferWrite,
                      vk::PipelineStageFlagBits2::eTransfer}})
        .SetExecution([this](vk::CommandBuffer cmd,
                             ExecutorConfig::ResourceStateInfos const& infos) {
            vk::BufferCopy2 copyRegion {};
            copyRegion.setSize(16);

            vk::CopyBufferInfo2 info {
                GetRenderResMgr()[infos[0].name].GetBufferHandle(),
                GetRenderResMgr()[infos[1].name].GetBufferHandle(), copyRegion};

            cmd.copyBuffer2(info);
        });

    cfg.AddRenderPass("DrawQuad", "QuadDraw")
        .SetBinding("tex", "DrawImage")
        .SetBinding("outFragColor", "_Swapchain_");

    cfg.Compile(sequence);
}
