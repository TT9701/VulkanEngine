#include "MeshShaderDemo.h"

#include <random>

using namespace IDNS_VC;

MeshShaderDemo::MeshShaderDemo(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescriptorSetPool(mContext.get())),
      mBindlessDescSetPool(CreateDescriptorSetPool(mContext.get(), 1 << 22)),
      mPrepassCopy(&mRenderResMgr),
      mBackgroundPass_PSO {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                           &mDescSetPool},
      mBackgroundPass_Barrier(mContext.get(), &mRenderResMgr),
      mMeshDrawPass {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                     &mDescSetPool},
      mMeshDrawPass_Barrier(mContext.get(), &mRenderResMgr),
      mMeshShaderPass {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                       &mDescSetPool},
      mMeshShaderPass_Barrier(mContext.get(), &mRenderResMgr),
      mQuadDrawPass_PSO {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                         &mDescSetPool, mSwapchain.get()},
      mQuadDrawPass_Barrier_Pre(mContext.get(), &mRenderResMgr,
                                mSwapchain.get()),
      mQuadDrawPass_Barrier_Post(mContext.get(), &mRenderResMgr,
                                 mSwapchain.get()),
      mGui(mContext.get(), mSwapchain.get(), mWindow.get()) {}

MeshShaderDemo::~MeshShaderDemo() = default;

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
                                    vk::ShaderStageFlagBits::eFragment, true);

    Type_ShaderMacros macros {};
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
    mGui.PollEvent(e);
}

void MeshShaderDemo::Update_OnResize() {
    Application::Update_OnResize();

    vk::Extent2D extent = {static_cast<uint32_t>(mWindow->GetWidth()),
                           static_cast<uint32_t>(mWindow->GetHeight())};

    mRenderResMgr.ResizeResources_ScreenSizeRelated(extent);

    auto resNames = mRenderResMgr.GetResourceNames_SrcreenSizeRelated();

    mBackgroundPass_PSO.Update(resNames);
    mBackgroundPass_Barrier.Update(resNames);

    // mMeshDrawPass.OnResize(extent);
    // mMeshDrawPass_Barrier.OnResize(extent);

    mMeshShaderPass.OnResize(extent);
    mMeshShaderPass_Barrier.Update(resNames);

    mQuadDrawPass_Barrier_Pre.Update(resNames);
    mQuadDrawPass_PSO.OnResize(extent);
    mQuadDrawPass_Barrier_Post.Update(resNames);
}

void MeshShaderDemo::UpdateScene() {
    Application::UpdateScene();

    auto view = mMainCamera.GetViewMatrix();

    glm::mat4 proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(mWindow->GetWidth())
                             / static_cast<float>(mWindow->GetHeight()),
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

    CreateDrawImage();
    CreateDepthImage();

    LoadShaders();
    CreatePipelines();

    CreateRandomTexture();

    mRenderResMgr.CreateBuffer(
        "SceneUniformBuffer", sizeof(SceneData),
        vk::BufferUsageFlagBits::eUniformBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::Staging);

    mRenderResMgr.CreateBuffer_ScreenSizeRelated(
        "RWBuffer",
        sizeof(glm::vec4) * mWindow->GetWidth() * mWindow->GetHeight(),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::DeviceLocal, sizeof(glm::vec4));

    mMainCamera.mPosition = glm::vec3 {0.0f, 1.0f, 2.0f};

    // cisdi model data converter
    {
        IntelliDesign_NS::Core::Utils::Timer timer;

        Type_STLString model = "sponza/sponza.obj";

        mFactoryModel = MakeShared<Geometry>(MODEL_PATH_CSTR(model));

        // mFactoryModel->GenerateBuffers(mContext.get(), this);
        mFactoryModel->GenerateMeshletBuffers(mContext.get(), this);

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
    //         mModels[i]->GenerateMeshletBuffers(mContext.get(), this);
    //     }
    //
    //     auto duration_LoadModel = timer.End();
    //     printf("Time consumed: %f s. \n", duration_LoadModel);
    // }

    RecordDrawBackgroundCmds();
    // RecordDrawMeshCmds();
    RecordMeshShaderDrawCmds();
    RecordDrawQuadCmds();

    PrepareUIContext();
}

void MeshShaderDemo::BeginFrame() {
    Application::BeginFrame();

    mGui.BeginFrame();
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

        mBackgroundPass_Barrier.RecordCmd(cmd.GetHandle());
        mBackgroundPass_PSO.RecordCmd(cmd.GetHandle());

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

        // mMeshDrawPass_Barrier.RecordCmd(cmd.GetHandle());
        // mMeshDrawPass.RecordCmd(cmd.GetHandle());

        mMeshShaderPass.RecordCmd(cmd.GetHandle());
        mMeshShaderPass_Barrier.RecordCmd(cmd.GetHandle());

        mQuadDrawPass_Barrier_Pre.Update("_Swapchain_");
        mQuadDrawPass_PSO.Update("_Swapchain_");
        mQuadDrawPass_Barrier_Post.Update("_Swapchain_");

        mQuadDrawPass_Barrier_Pre.RecordCmd(cmd.GetHandle());
        mQuadDrawPass_PSO.RecordCmd(cmd.GetHandle());
        mQuadDrawPass_Barrier_Post.RecordCmd(cmd.GetHandle());

        mGui.Draw(cmd.GetHandle());

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

    auto ptr = mRenderResMgr.CreateTexture_ScreenSizeRelated(
        "DrawImage", RenderResource::Type::Texture2D,
        vk::Format::eR16G16B16A16Sfloat, drawImageExtent, drawImageUsage);
    ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);
}

void MeshShaderDemo::CreateDepthImage() {
    vk::Extent3D depthImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                   static_cast<uint32_t>(mWindow->GetHeight()),
                                   1};

    vk::ImageUsageFlags depthImageUsage {};
    depthImageUsage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto ptr = mRenderResMgr.CreateTexture_ScreenSizeRelated(
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

void MeshShaderDemo::CreateRandomTexture() {
    auto extent = VkExtent3D {16, 16, 1};

    size_t dataSize = extent.width * extent.height * 4;

    auto uploadBuffer = mRenderResMgr.CreateBuffer(
        "staging", dataSize * 32, vk::BufferUsageFlagBits::eTransferSrc,
        Buffer::MemoryType::Staging);

    ::std::string baseName {"RandomImage"};

    for (uint32_t i = 0; i < 32; ++i) {
        std::random_device rndDevice;
        std::default_random_engine rndEngine(rndDevice());
        std::uniform_int_distribution rndDist(50, UCHAR_MAX);

        std::array<uint8_t, 16 * 16 * 4> pixels;
        for (uint32_t j = 0; j < 16 * 16; ++j) {
            pixels[j * 4] = rndDist(rndEngine);
            pixels[j * 4 + 1] = rndDist(rndEngine);
            pixels[j * 4 + 2] = rndDist(rndEngine);
            pixels[j * 4 + 3] = 255;
        }

        auto name = baseName + ::std::to_string(i);
        auto ptr = mRenderResMgr.CreateTexture(
            name.c_str(), RenderResource::Type::Texture2D,
            vk::Format::eR8G8B8A8Unorm, extent,
            vk::ImageUsageFlagBits::eSampled
                | vk::ImageUsageFlagBits::eTransferDst);
        ptr->CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

        memcpy((char*)uploadBuffer->GetBufferMappedPtr() + dataSize * i,
               pixels.data(), dataSize);
    }

    mImmSubmitMgr.Submit([&](vk::CommandBuffer cmd) {
        for (uint32_t i = 0; i < 32; ++i) {
            auto name = baseName + ::std::to_string(i);
            Utils::TransitionImageLayout(
                cmd, mRenderResMgr[name.c_str()]->GetTexHandle(),
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal);

            vk::BufferImageCopy2 copyRegion {};
            copyRegion.setBufferOffset(16 * 16 * 4 * i)
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(extent);

            mPrepassCopy.CopyBufferToImage("staging", name.c_str(), copyRegion);
        }

        mPrepassCopy.GenerateMetaData();

        mPrepassCopy.RecordCmd(cmd);

        for (uint32_t i = 0; i < 32; ++i) {
            auto name = baseName + ::std::to_string(i);
            Utils::TransitionImageLayout(
                cmd, mRenderResMgr[name.c_str()]->GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    });

    // create descriptor
    vk::DescriptorType descType {vk::DescriptorType::eCombinedImageSampler};
    auto descBufProps = mContext->GetDescBufProps();
    DescriptorSetLayout bindlessLayout {
        mContext.get(),
        {"tex"},
        {{0, descType, 1024 * 1024, vk::ShaderStageFlagBits::eFragment}},
        descBufProps,
        nullptr};

    mBindlessSet = MakeShared<DescriptorSet>(mContext.get(), &bindlessLayout);
    auto requestHandle =
        mBindlessDescSetPool.RequestUnit(bindlessLayout.GetSize());
    mBindlessSet->SetRequestedHandle(std::move(requestHandle));
    auto descSize = bindlessLayout.GetDescriptorSize(descType);

    auto resource = mBindlessSet->GetPoolResource();

    vk::DescriptorImageInfo imageInfo {};
    imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSampler(mContext->GetDefaultNearestSamplerHandle());

    for (uint32_t i = 0; i < 32; ++i) {
        auto name = baseName + ::std::to_string(i);
        imageInfo.setImageView(mRenderResMgr[name.c_str()]->GetTexViewHandle());
        vk::DescriptorGetInfoEXT descInfo {};
        descInfo.setType(descType).setData(&imageInfo).setPNext(nullptr);

        mContext->GetDeviceHandle().getDescriptorEXT(
            descInfo, descSize,
            (char*)resource.hostAddr + resource.offset
                + mBindlessSet->GetBingdingOffset(0) + i * descSize);
    }
}

void MeshShaderDemo::CreateBackgroundComputePipeline() {
    auto compute =
        mShaderMgr.GetShader("computeDraw", vk::ShaderStageFlagBits::eCompute);
    auto program = mShaderMgr.CreateProgram("background", compute);

    auto builder = mPipelineMgr.GetComputePipelineBuilder();

    auto backgroundComputePipeline =
        builder.SetShaderProgram(program)
            .SetFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
            .Build("Background");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void MeshShaderDemo::CreateMeshPipeline() {
    auto vert =
        mShaderMgr.GetShader("vertex", vk::ShaderStageFlagBits::eVertex);
    auto frag =
        mShaderMgr.GetShader("fragment", vk::ShaderStageFlagBits::eFragment);
    auto program = mShaderMgr.CreateProgram("mesh draw", vert, frag);

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder();
    builder.SetShaderProgram(program)
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
    Type_ShaderMacros macros {};
    macros.emplace("TASK_INVOCATION_COUNT",
                   std::to_string(TASK_SHADER_INVOCATION_COUNT));

    auto task = mShaderMgr.GetShader("Mesh shader task",
                                     vk::ShaderStageFlagBits::eTaskEXT, macros);

    macros.clear();
    macros.emplace("MESH_INVOCATION_COUNT",
                   std::to_string(MESH_SHADER_INVOCATION_COUNT));
    macros.emplace("MAX_VERTICES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_VERTICES));
    macros.emplace("MAX_PRIMITIVES",
                   std::to_string(NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES));

    auto mesh = mShaderMgr.GetShader("Mesh shader mesh",
                                     vk::ShaderStageFlagBits::eMeshEXT, macros);

    auto frag = mShaderMgr.GetShader("Mesh shader fragment",
                                     vk::ShaderStageFlagBits::eFragment);

    auto program = mShaderMgr.CreateProgram("meshlet draw", task, mesh, frag);

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder();
    builder.SetShaderProgram(program)
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
    auto vert =
        mShaderMgr.GetShader("Quad vertex", vk::ShaderStageFlagBits::eVertex);
    auto frag = mShaderMgr.GetShader("Quad fragment",
                                     vk::ShaderStageFlagBits::eFragment);

    auto program = mShaderMgr.CreateProgram("QuadDraw", vert, frag);

    auto builder = mPipelineMgr.GetGraphicsPipelineBuilder();
    builder.SetShaderProgram(program)
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
    auto width = mRenderResMgr["DrawImage"]->GetTexWidth();
    auto height = mRenderResMgr["DrawImage"]->GetTexHeight();

    // barrier
    {
        mBackgroundPass_Barrier.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
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
    auto width = mRenderResMgr["DrawImage"]->GetTexWidth();
    auto height = mRenderResMgr["DrawImage"]->GetTexHeight();

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
        mMeshDrawPass.SetPipeline("TriangleDraw");

        mMeshDrawPass["constants"] = RenderPassBinding::PushContants {
            sizeof(*pPushConstants), pPushConstants};

        mMeshDrawPass["SceneDataUBO"] = "SceneUniformBuffer";
        mMeshDrawPass["tex"] = "ErrorCheckImage";

        mMeshDrawPass["outFragColor"] = {"DrawImage", "Color-Whole"};

        mMeshDrawPass[RenderPassBinding::Type::DSV] = {"DepthImage",
                                                       "Depth-Whole"};

        mMeshDrawPass[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mMeshDrawPass.GenerateMetaData();
    }

    auto& dcMgr = mMeshDrawPass.GetDrawCallManager();

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_DrawIndiret(
        mFactoryModel->GetIndirectCmdBuffer()->GetHandle(), 0,
        mFactoryModel->GetMeshCount(), sizeof(vk::DrawIndirectCommand));
}

void MeshShaderDemo::RecordDrawQuadCmds() {
    auto width = mSwapchain->GetExtent2D().width;
    auto height = mSwapchain->GetExtent2D().height;

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
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mQuadDrawPass_PSO.GenerateMetaData();
    }
    auto& dcMgr = mQuadDrawPass_PSO.GetDrawCallManager();

    vk::Viewport viewport {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
    dcMgr.AddArgument_Viewport(0, {viewport});

    vk::Rect2D scissor {{0, 0}, {width, height}};
    dcMgr.AddArgument_Scissor(0, {scissor});

    dcMgr.AddArgument_Draw(3, 1, 0, 0);
}

void MeshShaderDemo::RecordMeshShaderDrawCmds() {
    uint32_t width = mRenderResMgr["DrawImage"]->GetTexWidth();
    uint32_t height = mRenderResMgr["DrawImage"]->GetTexHeight();

    auto meshPushContants = mFactoryModel->GetMeshletPushContantsPtr();
    // meshPushContants->mModelMatrix =
    //     glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.0001f});
    meshPushContants->mModelMatrix =
        glm::rotate(glm::scale(glm::mat4 {1.0f}, glm::vec3 {0.01f}),
                    glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

    // barrier
    {
        mMeshShaderPass_Barrier.AddImageBarrier(
            "DrawImage",
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader
                           | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
             .oldLayout = vk::ImageLayout::eGeneral,
             .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .aspect = vk::ImageAspectFlagBits::eColor});
        mMeshShaderPass_Barrier.GenerateMetaData();
    }

    // PSO
    {
        mMeshShaderPass.SetPipeline("MeshShaderDraw");

        mMeshShaderPass["PushConstants"] = RenderPassBinding::PushContants {
            sizeof(*meshPushContants), meshPushContants};

        mMeshShaderPass["UBO"] = "SceneUniformBuffer";

        auto bindlessSet = mBindlessSet->GetPoolResource();
        mMeshShaderPass["tex"] = RenderPassBinding::BindlessDescBufInfo {
            bindlessSet.deviceAddr, bindlessSet.offset};

        mMeshShaderPass["outFragColor"] = {"DrawImage", "Color-Whole"};
        mMeshShaderPass[RenderPassBinding::Type::DSV] = {"DepthImage",
                                                         "Depth-Whole"};

        mMeshShaderPass[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {{{0, 0}, {width, height}}, 1, 0};

        mMeshShaderPass.GenerateMetaData();
    }

    auto& dcMgr = mMeshShaderPass.GetDrawCallManager();

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
}

void MeshShaderDemo::UpdateSceneUBO() {
    auto data = mRenderResMgr["SceneUniformBuffer"]->GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

void MeshShaderDemo::PrepareUIContext() {
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
        }
        ImGui::End();
    });
}