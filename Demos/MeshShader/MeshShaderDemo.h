#pragma once

#include "Core/Application/Application.h"
#include "Core/Application/EntryPoint.h"

#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

#include "Core/SceneGraph/Node.h"
#include "Core/SceneGraph/Scene.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"

namespace IDVC_NS = IntelliDesign_NS::Vulkan::Core;
namespace IDCMP_NS = IntelliDesign_NS::Core::MemoryPool;

namespace IDC_NS = IntelliDesign_NS::Core;
namespace IDCSG_NS = IDC_NS::SceneGraph;

struct SceneData {
    IDCMCore_NS::Float4 sunLightPos {0.6f, 1.0f, 0.8f, 1.0f};
    IDCMCore_NS::Float4 sunLightColor {1.0f, 1.0f, 1.0f, 1.0f};
    IDCMCore_NS::Float4 cameraPos {};
    IDCMCore_NS::Mat4 view {};
    IDCMCore_NS::Mat4 proj {};
    IDCMCore_NS::Mat4 viewProj {};

    IDCMCore_NS::Float4 objColor {0.7f, 0.7f, 0.7f, 1.0f};
    IDCMCore_NS::Float4 metallicRoughness {0.5f, 0.0f, 0.0f, 0.0f};
    int32_t texIndex {0};
};

class MeshShaderDemo : public IDVC_NS::Application {
public:
    MeshShaderDemo(IDVC_NS::ApplicationSpecification const& spec);

    ~MeshShaderDemo() override;

private:
    virtual void CreatePipelines() override;
    virtual void LoadShaders() override;
    virtual void PollEvents(SDL_Event* e, float deltaTime) override;
    virtual void Update_OnResize() override;
    virtual void UpdateScene() override;
    virtual void Prepare() override;

    virtual void BeginFrame(IDVC_NS::RenderFrame& frame) override;
    virtual void RenderFrame(IDVC_NS::RenderFrame& frame) override;
    virtual void EndFrame(IDVC_NS::RenderFrame& frame) override;

    virtual void RenderToSwapchainBindings(vk::CommandBuffer cmd) override;

private:
    void CreateDrawImage();
    void CreateDepthImage();
    void CreateRandomTexture();

    void CreateBackgroundComputePipeline();
    void CreateMeshShaderPipeline();
    void CreateDrawQuadPipeline();

    void UpdateSceneUBO();

    void PrepareUIContext();

    void RecordPasses(IDVC_NS::RenderSequence& sequence);

private:
    IDVC_NS::DescriptorSetPool mDescSetPool;

    IDVC_NS::RenderSequence mRenderSequence;

    IDVC_NS::Semaphore mCopySem;
    IDVC_NS::Semaphore mCmpSem;

    IDVC_NS::SharedPtr<IDVC_NS::Buffer> mDispatchIndirectCmdBuffer {};

    IDVC_NS::UniquePtr<IDC_NS::Camera> mMainCamera {};
    SceneData mSceneData {};

    IDVC_NS::UniquePtr<IntelliDesign_NS::ModelData::ModelDataManager>
        mModelMgr {};
    IDVC_NS::UniquePtr<IDVC_NS::GPUGeometryDataManager> mGeoMgr {};

    IDVC_NS::SharedPtr<IDCSG_NS::Scene> mScene {};

    IDVC_NS::Type_STLString mImageName0 {};
    IDVC_NS::Type_STLString mImageName1 {};
};

VE_CREATE_APPLICATION(MeshShaderDemo, 1600, 900);