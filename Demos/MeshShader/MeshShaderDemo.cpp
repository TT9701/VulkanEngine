#include "MeshShaderDemo.h"

#include <random>

using namespace IDNS_VC;

BindlessDescPool::BindlessDescPool(
    Context* context,
    IDNC_CMP::Type_STLVector<IDNS_VC::RenderPassBindingInfo_PSO*> const& pso,
    vk::DescriptorType type)
    : pContext(context), mPSOs(pso), mDescType(type) {
    auto descBufProps = pContext->GetDescBufProps();
    mDescCount =
        ::std::min(MAX_BINDLESS_DESCRIPTOR_COUNT,
                   pContext->GetDescIndexingProps()
                       .maxPerStageDescriptorUpdateAfterBindSampledImages);

    mLayout = MakeShared<DescriptorSetLayout>(
        pContext, Type_STLVector<Type_STLString> {"sceneTexs"},
        Type_STLVector<vk::DescriptorSetLayoutBinding> {
            vk::DescriptorSetLayoutBinding {
                0,
                mDescType,
                mDescCount,
                vk::ShaderStageFlagBits::eFragment,
            }},
        descBufProps, nullptr);

    mDescSize = mLayout->GetDescriptorSize(mDescType);
    mDescSetPool = MakeDescSetPoolPtr(pContext, mDescSize * mDescCount);

    mSet = MakeShared<DescriptorSet>(pContext, mLayout.get());
    auto requestHandle = mDescSetPool->RequestUnit(mLayout->GetSize());
    mSet->SetRequestedHandle(std::move(requestHandle));
}

PoolResource BindlessDescPool::GetPoolResource() const {
    return mSet->GetPoolResource();
}

uint32_t BindlessDescPool::Add(Texture const* texture) {
    if (mDescIndexMap.contains(texture))
        return mDescIndexMap.at(texture);

    uint32_t idx;
    bool success = mAvailableIndices.try_dequeue(idx);
    if (!success) {
        if (mCurrentDescCount < mDescCount) {
            idx = mCurrentDescCount++;
        } else {
            ExpandSet();
            return Add(texture);
        }
    }

    auto resource = mSet->GetPoolResource();

    vk::DescriptorImageInfo imageInfo {};
    imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSampler(pContext->GetDefaultNearestSamplerHandle());

    imageInfo.setImageView(texture->GetViewHandle());
    vk::DescriptorGetInfoEXT descInfo {};
    descInfo.setType(mDescType).setData(&imageInfo).setPNext(nullptr);

    pContext->GetDeviceHandle().getDescriptorEXT(
        descInfo, mDescSize,
        (char*)resource.hostAddr + resource.offset + mSet->GetBingdingOffset(0)
            + idx * mDescSize);

    mDescIndexMap.emplace(texture, idx);

    return idx;
}

uint32_t BindlessDescPool::Delete(IDNS_VC::Texture const* texture) {
    uint32_t idx;
    if (mDescIndexMap.contains(texture)) {
        idx = mDescIndexMap.at(texture);

        auto resource = mSet->GetPoolResource();
        auto descSize = mLayout->GetDescriptorSize(mDescType);
        memset((char*)resource.hostAddr + resource.offset
                   + mSet->GetBingdingOffset(0) + idx * descSize,
               0, descSize);

        mAvailableIndices.enqueue(idx);
        mDescIndexMap.erase(texture);
    } else {
        throw ::std::runtime_error(
            "texture is not contained in bindless pool.");
    }
    return idx;
}

void BindlessDescPool::ExpandSet() {
    mDescCount *= 2;
    auto origSize = mLayout->GetSize();

    mLayout.reset();
    mLayout = MakeShared<DescriptorSetLayout>(
        pContext, Type_STLVector<Type_STLString> {"sceneTexs"},
        Type_STLVector<vk::DescriptorSetLayoutBinding> {
            vk::DescriptorSetLayoutBinding {
                0,
                mDescType,
                mDescCount,
                vk::ShaderStageFlagBits::eFragment,
            }},
        pContext->GetDescBufProps(), nullptr);

    auto requestHandle = mDescSetPool->RequestUnit(origSize * 2);
    auto resource = requestHandle.Get_Resource();
    memcpy(resource.hostAddr, mSet->GetPoolResource().hostAddr, origSize);

    mSet.reset();
    mSet = MakeShared<DescriptorSet>(pContext, mLayout.get());
    mSet->SetRequestedHandle(std::move(requestHandle));

    for (auto& pso : mPSOs) {
        pso->Update("sceneTexs", RenderPassBinding::BindlessDescBufInfo {
                                     resource.deviceAddr, resource.offset});
    }
}

void FrameResourceManager::BuildFrameResources(
    Context* context,
    IDNC_CMP::Type_STLVector<IDNS_VC::RenderPassBindingInfo_PSO*> const& pso) {
    mFrameResources.resize(FRAME_OVERLAP);
    auto ptr = MakeShared<BindlessDescPool>(context, pso);
    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        mFrameResources[i].mBindlessDescPool = ptr;
    }
}

FrameResource* FrameResourceManager::GetCurrentFrameResource(
    uint32_t frameIdx) {
    return &mFrameResources[frameIdx % FRAME_OVERLAP];
}

MeshShaderDemo::MeshShaderDemo(ApplicationSpecification const& spec)
    : Application(spec),
      mDescSetPool(CreateDescSetPool(mContext.get())),
      // mBindlessDescPool(mContext.get()),
      mPrepassCopy(&mRenderResMgr),
      mBackgroundPass_PSO {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                           &mDescSetPool},
      mBackgroundPass_Barrier(mContext.get(), &mRenderResMgr),
      mMeshDrawPass_PSO {mContext.get(), &mRenderResMgr, &mPipelineMgr,
                         &mDescSetPool},
      mMeshDrawPass_Barrier(mContext.get(), &mRenderResMgr),
      mMeshShaderPass_PSO {mContext.get(), &mRenderResMgr, &mPipelineMgr,
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

    // mMeshDrawPass_PSO.OnResize(extent);
    // mMeshDrawPass_Barrier.OnResize(extent);

    mMeshShaderPass_PSO.OnResize(extent);
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

    mFrameResMgr.BuildFrameResources(mContext.get(), {&mMeshShaderPass_PSO});
    CreateRandomTexture();

    CreateDrawImage();
    CreateDepthImage();

    LoadShaders();
    CreatePipelines();

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
        // mMeshDrawPass_PSO.RecordCmd(cmd.GetHandle());

        mMeshShaderPass_PSO.RecordCmd(cmd.GetHandle());
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
    auto extent = vk::Extent3D {16, 16, 1};
    const uint32_t randomImageCount = 128;

    size_t dataSize = extent.width * extent.height * 4;

    auto uploadBuffer = mRenderResMgr.CreateBuffer(
        "staging", dataSize * randomImageCount,
        vk::BufferUsageFlagBits::eTransferSrc, Buffer::MemoryType::Staging);

    ::std::string baseName {"RandomImage"};

    for (uint32_t i = 0; i < randomImageCount; ++i) {
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
        for (uint32_t i = 0; i < randomImageCount; ++i) {
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

        for (uint32_t i = 0; i < randomImageCount; ++i) {
            auto name = baseName + ::std::to_string(i);
            Utils::TransitionImageLayout(
                cmd, mRenderResMgr[name.c_str()]->GetTexHandle(),
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    });

    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        for (uint32_t j = 0; j < randomImageCount / 2; ++j) {
            auto name = baseName + ::std::to_string(j);
            auto texture = mRenderResMgr[name.c_str()]->GetTexPtr();

            mFrameResMgr.GetCurrentFrameResource(mFrameNum)[i]
                .mBindlessDescPool->Add(texture);
        }
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
        mMeshShaderPass_PSO.SetPipeline("MeshShaderDraw");

        mMeshShaderPass_PSO["PushConstants"] = RenderPassBinding::PushContants {
            sizeof(*meshPushContants), meshPushContants};

        mMeshShaderPass_PSO["UBO"] = "SceneUniformBuffer";

        auto bindlessSet = mFrameResMgr.GetCurrentFrameResource(mFrameNum)
                               ->mBindlessDescPool->GetPoolResource();
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
}

void MeshShaderDemo::UpdateSceneUBO() {
    auto data = mRenderResMgr["SceneUniformBuffer"]->GetBufferMappedPtr();
    memcpy(data, &mSceneData, sizeof(mSceneData));
}

void MeshShaderDemo::PrepareUIContext() {
    mImageName0 = "RandomImage";
    mImageName1 = "RandomImage";
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
                auto tex = mRenderResMgr[mImageName0.c_str()]->GetTexPtr();
                auto idx = mFrameResMgr.GetCurrentFrameResource(mFrameNum)
                               ->mBindlessDescPool->Add(tex);

                DBG_LOG_INFO("Add Button pressed, add texture at idx %d", idx);
            }

            ImGui::InputText("###", mImageName1.data(), 32);
            ImGui::SameLine();
            if (ImGui::Button("Delete")) {
                auto tex = mRenderResMgr[mImageName1.c_str()]->GetTexPtr();
                auto idx = mFrameResMgr.GetCurrentFrameResource(mFrameNum)
                               ->mBindlessDescPool->Delete(tex);
                DBG_LOG_INFO("Delete Button pressed, delete texture at idx %d",
                             idx);
            }
        }
        ImGui::End();
    });
}