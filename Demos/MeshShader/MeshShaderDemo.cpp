#include "MeshShaderDemo.hpp"

MeshShaderDemo::MeshShaderDemo(IDNS_VC::ApplicationSpecification const& spec)
    : Application(spec),
      mMeshShaderPass {&mRenderResMgr, &mPipelineMgr, &mDescMgr},
      mBackgroundDrawCallMgr {&mRenderResMgr},
      mMeshDrawCallMgr {&mRenderResMgr},
      mQuadDrawCallMgr {&mRenderResMgr} {}

MeshShaderDemo::~MeshShaderDemo() {}

void MeshShaderDemo::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
    CreateMeshShaderPipeline();
}

void MeshShaderDemo::LoadShaders() {
    mShaderMgr.CreateShaderFromGLSL("computeDraw",
                                    SHADER_PATH_CSTR("BackGround.comp"),
                                    vk::ShaderStageFlagBits::eCompute);

    mShaderMgr.CreateShaderFromGLSL("vertex", SHADER_PATH_CSTR("Triangle.vert"),
                                    vk::ShaderStageFlagBits::eVertex, true);

    mShaderMgr.CreateShaderFromGLSL("fragment",
                                    SHADER_PATH_CSTR("Triangle.frag"),
                                    vk::ShaderStageFlagBits::eFragment, true);

    mShaderMgr.CreateShaderFromGLSL("Mesh shader fragment",
                                    SHADER_PATH_CSTR("MeshShader.frag"),
                                    vk::ShaderStageFlagBits::eFragment);

    IDNS_VC::Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));
    mShaderMgr.CreateShaderFromGLSL(
        "Mesh shader task", SHADER_PATH_CSTR("MeshShader.task"),
        vk::ShaderStageFlagBits::eTaskEXT, false, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));
    mShaderMgr.CreateShaderFromGLSL(
        "Mesh shader mesh", SHADER_PATH_CSTR("MeshShader.mesh"),
        vk::ShaderStageFlagBits::eMeshEXT, true, macros);

    mShaderMgr.CreateShaderFromGLSL("Quad vertex",
                                    SHADER_PATH_CSTR("Quad.vert"),
                                    vk::ShaderStageFlagBits::eVertex);

    mShaderMgr.CreateShaderFromGLSL("Quad fragment",
                                    SHADER_PATH_CSTR("Quad.frag"),
                                    vk::ShaderStageFlagBits::eFragment);
}

void MeshShaderDemo::PollEvents(SDL_Event* e, float deltaTime) {
    Application::PollEvents(e, deltaTime);

    mMainCamera.ProcessSDLEvent(e, deltaTime);
}

void MeshShaderDemo::Update_OnResize() {
    Application::Update_OnResize();

    vk::Extent2D extent = {static_cast<uint32_t>(mWindow->GetWidth()),
                           static_cast<uint32_t>(mWindow->GetHeight())};

    mRenderResMgr.ResizeScreenSizeRelatedResources(extent);

    mBackgroundDrawCallMgr.UpdateArgument_OnResize(extent);
    // mMeshDrawCallMgr.UpdateArgument_OnResize(extent);
    mMeshShaderPass.GetDrawCallManager().UpdateArgument_OnResize(extent);
    mQuadDrawCallMgr.UpdateArgument_OnResize(extent);
}

void MeshShaderDemo::UpdateScene() {
    Application::UpdateScene();

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

void MeshShaderDemo::Prepare() {
    Application::Prepare();

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
            IDNS_VC::Buffer::MemoryType::Staging);

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
        mRenderResMgr.CreateScreenSizeRelatedBuffer(
            "RWBuffer",
            sizeof(glm::vec4) * mWindow->GetWidth() * mWindow->GetHeight(),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            IDNS_VC::Buffer::MemoryType::DeviceLocal, sizeof(glm::vec4));

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

        auto meshes = IDNS_VC::CISDI_3DModelDataConverter::LoadCISDIModelData(
            cisdiModelPath.c_str());

        mFactoryModel = IDNS_VC::MakeShared<IDNS_VC::Model>(meshes);

        // mFactoryModel->GenerateBuffers(mContext.get(), this);
        mFactoryModel->GenerateMeshletBuffers(mContext.get(), this);
    }

    RecordDrawBackgroundCmds();
    // RecordDrawMeshCmds();
    RecordMeshShaderDrawCmds();
    RecordDrawQuadCmds();
}

void MeshShaderDemo::BeginFrame() {
    Application::BeginFrame();
}

void MeshShaderDemo::RenderFrame() {
    auto scIdx = mSwapchain->GetCurrentImageIndex();

    const uint64_t graphicsFinished =
        mContext->GetTimelineSemphore()->GetValue();
    const uint64_t computeFinished = graphicsFinished + 1;
    const uint64_t allFinished = graphicsFinished + 2;

    // Compute Draw
    {
        auto cmd = mCmdMgr.GetCmdBufferToBegin();

        mBackgroundDrawCallMgr.RecordCmd(cmd.GetHandle());

        cmd.End();

        IDNS_VC::Type_STLVector<IDNS_VC::SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             mSwapchain->GetReady4RenderSemHandle(), 0ui64},
            {vk::PipelineStageFlagBits2::eBottomOfPipe,
             mContext->GetTimelineSemaphoreHandle(), graphicsFinished}};

        IDNS_VC::Type_STLVector<IDNS_VC::SemSubmitInfo> signals = {
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
        mMeshShaderPass.GetDrawCallManager().RecordCmd(cmd.GetHandle());

        mQuadDrawCallMgr.UpdateArgument_Attachments(
            {0}, {mSwapchain->GetColorAttachmentInfo(scIdx)});
        mQuadDrawCallMgr.UpdateArgument_Barriers_BeforePass(
            {"Swapchain"}, {mSwapchain->GetImageBarrier_BeforePass(scIdx)}, {},
            {});
        mQuadDrawCallMgr.UpdateArgument_Barriers_AfterPass(
            {"Swapchain"}, {mSwapchain->GetImageBarrier_AfterPass(scIdx)}, {},
            {});

        mQuadDrawCallMgr.RecordCmd(cmd.GetHandle());

        cmd.End();

        IDNS_VC::Type_STLVector<IDNS_VC::SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eComputeShader,
             mContext->GetTimelineSemaphoreHandle(), computeFinished}};

        IDNS_VC::Type_STLVector<IDNS_VC::SemSubmitInfo> signals = {
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

        IDNS_VC::Type_STLVector<IDNS_VC::SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             mContext->GetTimelineSemaphoreHandle(), allFinished + 1}};

        mCmdMgr.Submit(cmd.GetHandle(),
                       mContext->GetDevice()->GetGraphicQueue(), {}, signals);
    }
}

void MeshShaderDemo::EndFrame() {
    Application::EndFrame();
}

void MeshShaderDemo::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                  static_cast<uint32_t>(mWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    auto ptr = mRenderResMgr.CreateScreenSizeRelatedTexture(
        "DrawImage", IDNS_VC::RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, drawImageExtent, drawImageUsage);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
}

void MeshShaderDemo::CreateDepthImage() {
    vk::Extent3D depthImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                   static_cast<uint32_t>(mWindow->GetHeight()),
                                   1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto ptr = mRenderResMgr.CreateScreenSizeRelatedTexture(
        "DepthImage", IDNS_VC::RenderResource::Type::Texture2D,
        vk::Format::eD24UnormS8Uint, depthImageExtent, depthImageUsage);
    ptr->CreateTexView("Depth-Whole", vk::ImageAspectFlagBits::eDepth
                                          | vk::ImageAspectFlagBits::eStencil);

    mImmSubmitMgr.Submit([&](vk::CommandBuffer cmd) {
        IDNS_VC::Utils::TransitionImageLayout(
            cmd, ptr->GetTexHandle(), vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    });
}

void MeshShaderDemo::CreateErrorCheckTexture() {
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
        "ErrorCheckImage", IDNS_VC::RenderResource::Type::Texture2D,
        vk::Format::eR8G8B8A8Unorm, extent,
        vk::ImageUsageFlagBits::eSampled
            | vk::ImageUsageFlagBits::eTransferDst);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

    {
        size_t dataSize = extent.width * extent.height * 4;

        auto uploadBuffer = mContext->CreateStagingBuffer("", dataSize);
        memcpy(uploadBuffer->GetMapPtr(), pixels.data(), dataSize);

        mImmSubmitMgr.Submit([&](vk::CommandBuffer cmd) {
            IDNS_VC::Utils::TransitionImageLayout(
                cmd, ptr->GetTexHandle(), vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal);

            vk::BufferImageCopy copyRegion {};
            copyRegion
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(extent);

            cmd.copyBufferToImage(
                uploadBuffer->GetHandle(), ptr->GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal, copyRegion);

            IDNS_VC::Utils::TransitionImageLayout(
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

void MeshShaderDemo::CreateBackgroundComputePipeline() {
    auto builder = mPipelineMgr.GetComputePipelineBuilder(&mDescMgr);

    auto backgroundComputePipeline =
        builder
            .SetShader(mShaderMgr.GetShader("computeDraw",
                                            vk::ShaderStageFlagBits::eCompute))
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void MeshShaderDemo::CreateMeshPipeline() {
    IDNC_CMP::Type_STLVector<IDNS_VC::Shader*> shaders;
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

void MeshShaderDemo::CreateMeshShaderPipeline() {
    IDNC_CMP::Type_STLVector<IDNS_VC::Shader*> shaders;
    shaders.reserve(3);
    shaders.emplace_back(mShaderMgr.GetShader(
        "Mesh shader fragment", vk::ShaderStageFlagBits::eFragment));

    IDNS_VC::Type_ShaderMacros macros {};
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

void MeshShaderDemo::CreateDrawQuadPipeline() {
    IDNC_CMP::Type_STLVector<IDNS_VC::Shader*> shaders;
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

void MeshShaderDemo::RecordDrawBackgroundCmds() {
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
        IDNS_VC::Utils::GetWholeImageSubresource(
            vk::ImageAspectFlagBits::eColor)};
    mBackgroundDrawCallMgr.AddArgument_Barriers_BeforePass({"DrawImage"},
                                                           {drawImageBarrier});

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));

    vk::ClearColorValue clearValue {flash, flash, flash, 1.0f};

    auto subresource = vk::ImageSubresourceRange {
        vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0,
        vk::RemainingArrayLayers};

    mBackgroundDrawCallMgr.AddArgument_ClearColorImage(
        "DrawImage", vk::ImageLayout::eGeneral, clearValue, {subresource});

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
}

void MeshShaderDemo::RecordDrawMeshCmds() {
    auto width = mRenderResMgr["DrawImage"]->GetTexWidth();
    auto height = mRenderResMgr["DrawImage"]->GetTexHeight();

    {
        vk::Viewport viewport {0.0f,          0.0f, (float)width,
                               (float)height, 0.0f, 1.0f};
        mMeshDrawCallMgr.AddArgument_Viewport(0, {viewport});

        vk::Rect2D scissor {{0, 0}, {width, height}};
        mMeshDrawCallMgr.AddArgument_Scissor(0, {scissor});
    }

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
        IDNS_VC::Utils::GetWholeImageSubresource(
            vk::ImageAspectFlagBits::eColor)};
    mMeshDrawCallMgr.AddArgument_Barriers_BeforePass({"DrawImage"},
                                                     {drawImageBarrier});

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
        {{0, 0}, {width, height}}, 1, 0,
        {{"DrawImage", "Color-Whole", colorAttachment}},
        {"DepthImage", "Depth-Whole", depthAttachment});

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

void MeshShaderDemo::RecordDrawQuadCmds() {
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
        IDNS_VC::Utils::GetWholeImageSubresource(
            vk::ImageAspectFlagBits::eColor)};

    auto scBarrier = mSwapchain->GetImageBarrier_BeforePass(
        mSwapchain->GetCurrentImageIndex());

    mQuadDrawCallMgr.AddArgument_Barriers_BeforePass(
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

    mQuadDrawCallMgr.AddArgument_Barriers_AfterPass({"Swapchain"}, {scBarrier});
}

void MeshShaderDemo::RecordMeshShaderDrawCmds() {
    uint32_t width = mRenderResMgr["DrawImage"]->GetTexWidth();
    uint32_t height = mRenderResMgr["DrawImage"]->GetTexHeight();

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    meshPushContants->mModelMatrix =
        glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});

    auto& dcMgr = mMeshShaderPass.GetDrawCallManager();
    {
        mMeshShaderPass.Init("MeshShaderDraw");

        mMeshShaderPass["_DescriptorBuffer_"] = {0};

        mMeshShaderPass["_PushContants_"] =
            IDNS_VC::RenderPassBinding::PushContants {sizeof(*meshPushContants),
                                                      meshPushContants};

        mMeshShaderPass.GenerateMetaData();
    }

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
        IDNS_VC::Utils::GetWholeImageSubresource(
            vk::ImageAspectFlagBits::eColor)};
    dcMgr.AddArgument_Barriers_BeforePass({"DrawImage"}, {drawImageBarrier});

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

    dcMgr.AddArgument_RenderingInfo(
        {{0, 0}, {width, height}}, 1, 0,
        {{"DrawImage", "Color-Whole", colorAttachment}},
        {"DepthImage", "Depth-Whole", depthAttachment});

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    // mMeshShaderDrawCallMgr.AddArgument_Pipeline(
    //     vk::PipelineBindPoint::eGraphics,
    //     mPipelineMgr.GetGraphicsPipelineHandle("MeshShaderDraw"));

    // mMeshShaderDrawCallMgr.AddArgument_DescriptorBuffer(
    //     {mDescMgr.GetDescBufferAddress(0)});

    auto sceneDataOffset =
        mDescMgr.GetDescriptorSet("MeshShader_Scene_Data")->GetOffsetInBuffer();
    dcMgr.AddArgument_DescriptorSet(
        vk::PipelineBindPoint::eGraphics,
        mPipelineMgr.GetLayoutHandle("MeshShaderDraw"), 0ui32, {0},
        {sceneDataOffset});

    // auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    // meshPushContants->mModelMatrix =
    //     glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});
    //
    // mMeshShaderDrawCallMgr.AddArgument_PushConstant(
    //     mPipelineMgr.GetLayoutHandle("MeshShaderDraw"),
    //     vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(*meshPushContants),
    //     meshPushContants);

    dcMgr.AddArgument_DrawMeshTasksIndirect(
        mFactoryModel->GetMeshTaskIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshes().size(),
        sizeof(vk::DrawMeshTasksIndirectCommandEXT));
}

void MeshShaderDemo::UpdateSceneUBO() {
    auto data = mRenderResMgr["SceneUniformBuffer"]->GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}