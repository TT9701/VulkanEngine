#pragma once

#include "Core/Application/Application.h"
#include "Core/Application/EntryPoint.h"

#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

#include "Core/SceneGraph/Node.h"
#include "Core/SceneGraph/Scene.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"

#include "Core/System/IDDeferredResourcePool.hpp"

namespace IDVC_NS = IntelliDesign_NS::Vulkan::Core;
namespace IDCMP_NS = IntelliDesign_NS::Core::MemoryPool;

namespace IDC_NS = IntelliDesign_NS::Core;
namespace IDCSG_NS = IDC_NS::SceneGraph;
namespace IDCMCore_NS = IntelliDesign_NS::CMCore_NS;

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

class DynamicLoading : public IDVC_NS::Application {
public:
    explicit DynamicLoading(IDVC_NS::ApplicationSpecification const& spec);

    ~DynamicLoading() override;

private:
    void CreatePipelines() override;
    void LoadShaders() override;
    void PollEvents(SDL_Event* e, float deltaTime) override;
    void Update_OnResize() override;
    void UpdateScene() override;
    void Prepare() override;

    void BeginFrame(IDVC_NS::RenderFrame& frame) override;
    void RenderFrame(IDVC_NS::RenderFrame& frame) override;
    void EndFrame(IDVC_NS::RenderFrame& frame) override;

    void RenderToSwapchainBindings(vk::CommandBuffer cmd) override;

private:
    void CreateDrawImage();
    void CreateDepthImage();

    void CreateBackgroundComputePipeline();
    void CreateMeshShaderPipeline();
    void CreateDrawQuadPipeline();

    void UpdateSceneUBO();

    void PrepareUIContext();

    void RecordPasses(IDVC_NS::RenderSequence& sequence,
                      IDVC_NS::RenderFrame& frame);

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

    using PrepareDGCDrawCommandSequenceTemp =
        DGCSeqTemplate<true, DGCExecutionSetType::None>;
    void prepare_dgc_draw_command();

    /**
     *  dgc dispath without execution set
     */
    void prepare_compute_sequence();

    IDCMCore_NS::Float3 _baseColorFactor {0.0f, 0.0f, 0.01f};

    using DispatchSequenceTemp =
        DGCSeqTemplate<true, DGCExecutionSetType::None, IDCMCore_NS::Float3>;

    /**
     *  dgc draw mesh task
     */
    void prepare_draw_mesh_task();

    using DrawSequenceTemp = DGCSeqTemplate<false, DGCExecutionSetType::None,
                                            IDVC_NS::MeshletPushConstants>;

    void ResizeToFitAllSeqBufPool(IDVC_NS::RenderFrame& frame);

    float AddNewNode(const char* modelPath);

    void RemoveNode(const char* nodeName);

    ::std::promise<IDVC_NS::SharedPtr<IDCSG_NS::NodeProxy<DrawSequenceTemp>>>
        mPromise;

    void UpdateFrustumCullingUBO();
};

VE_CREATE_APPLICATION(DynamicLoading, 1600, 900);