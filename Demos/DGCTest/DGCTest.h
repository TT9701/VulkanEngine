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

class DGCTest : public IDVC_NS::Application {
public:
    explicit DGCTest(IDVC_NS::ApplicationSpecification const& spec);

    ~DGCTest() override;

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


    using PrepareDGCDrawCommandSequenceTemp = DGCSeqTemplate<true, DGCExecutionSetType::None>;
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

    /**
     *  dgc dispath
     */
    void prepare_compute_sequence_pipeline();

    using DispatchSequence_PipelineTemp =
        DGCSeqTemplate<true, DGCExecutionSetType::Pipeline,
                       IDCMCore_NS::Float3>;

    /**
     *  dgc draw mesh task
     */
    void prepare_draw_mesh_task_pipeline();

    using DrawSequence_PipelineTemp =
        DGCSeqTemplate<false, DGCExecutionSetType::Pipeline,
                       IDVC_NS::MeshletPushConstants>;

    /**
     * ShaderEXT compute test
     */
    using DispatchSequence_ShaderTemp =
        DGCSeqTemplate<true, DGCExecutionSetType::Shader_Dispatch,
                       IDCMCore_NS::Float3>;

    void prepare_compute_sequence_shader();

    /**
     * ShaderEXT draw test
     */
    using DrawSequence_ShaderTemp =
        DGCSeqTemplate<false, DGCExecutionSetType::Shader_Draw,
                       IDVC_NS::MeshletPushConstants>;

    void prepare_draw_mesh_task_shader();

    const char* model0 {nullptr};
    const char* model1 {nullptr};
};

VE_CREATE_APPLICATION(DGCTest, 1600, 900);