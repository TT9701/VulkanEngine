#pragma once

#include "Core/Application/Application.h"
#include "Core/Application/EntryPoint.h"

#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

#include "Core/SceneGraph/Node.h"
#include "Core/SceneGraph/Scene.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"

#include "Core/Vulkan/Native/DGCSequence.h"

namespace IDVC_NS = IntelliDesign_NS::Vulkan::Core;
namespace IDCMP_NS = IntelliDesign_NS::Core::MemoryPool;

namespace IDC_NS = IntelliDesign_NS::Core;
namespace IDCSG_NS = IDC_NS::SceneGraph;
namespace IDCMCore_NS = IntelliDesign_NS::CMCore_NS;

// struct ComputeSequence {
//     uint32_t pipelineIdx;
//     IDCMCore_NS::Float3 pcBaseColorFactor;
//     vk::DispatchIndirectCommand dispatchCommand;
// };

// struct MeshTaskDrawSequence {
//     uint32_t pipelineIdx;
//     IDVC_NS::MeshletPushConstants meshletConstants;
//     IDVC_NS::FragmentPushConstants fragmentConstants;
//     vk::DrawIndirectCountIndirectCommandEXT drawCommand;
// };

struct DispatchSequence2
    : DGCSequenceTemplate<true, DGCExecutionSetType::Pipeline,
                          IDCMCore_NS::Float3> {};

struct DispatchSequence3 : DispatchSequence2 {};

// using DispatchSequence2 = SequenceTemplate<true, true, IDCMCore_NS::Float3>;

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

    /**
     *  dgc dispath
     */
    void prepare_compute_sequence();
    // uint32_t _computeSequenceSize;
    // IDVC_NS::SharedPtr<IDVC_NS::Buffer> _background_cmds;
    // vk::IndirectCommandsLayoutEXT _background_dgc_cmds_layout;
    uint32_t _preprocessSize;
    vk::Buffer _preprocessBuffer;
    vk::DeviceMemory _preprocessBufferMem;
    vk::DeviceAddress _preprocessBufAddr;

    void dgc_dispatch(vk::CommandBuffer cmd);

    IDCMCore_NS::Float3 _baseColorFactor {0.0f, 0.0f, 0.01f};
    // IDVC_NS::SharedPtr<IDVC_NS::Buffer> _dispatch_pc_buffer;
    // uint32_t _dispatch_sequenceCount {2};
    // uint32_t _dispatch_count {0};
    IDVC_NS::SharedPtr<IDVC_NS::Buffer> _readbackBuf;

    // vk::IndirectExecutionSetEXT _dispatch_executionSet;

    /**
     *  dgc draw mesh task
     */
    void prepare_draw_mesh_task();
    void dgc_draw_mesh_task(vk::CommandBuffer cmd);
    // uint32_t _drawSequenceSize;
    IDVC_NS::SharedPtr<IDVC_NS::Buffer> _draw_mesh_task_cmds;
    // vk::IndirectCommandsLayoutEXT _draw_mesh_task_dgc_cmds_layout;
    // size_t _draw_mesh_task_sequenceCount {2};
    // uint32_t _draw_mesh_task_draw_count {2000};
    // vk::IndirectExecutionSetEXT _draw_mesh_task_executionSet;

    uint32_t _draw_preprocessSize;
    vk::Buffer _draw_preprocessBuffer;
    vk::DeviceMemory _draw_preprocessBufferMem;
    vk::DeviceAddress _draw_preprocessBufAddr;

    /**
     * DGCSequenceLayout struct test
     */
    using DispatchSequence = IDVC_NS::DGCSequence<DGCSequenceTemplate<
        true, DGCExecutionSetType::Pipeline, IDCMCore_NS::Float3>>;

    using DrawSequence = IDVC_NS::DGCSequence<DGCSequenceTemplate<
        false, DGCExecutionSetType::Pipeline, IDVC_NS::MeshletPushConstants>>;

    IDVC_NS::UniquePtr<DispatchSequence> mDispatchSequence {};

    IDVC_NS::UniquePtr<DrawSequence> mDrawSequence {};

    /**
     * ShaderEXT compute test
     */
    using DispatchSequence_Shader = IDVC_NS::DGCSequence<DGCSequenceTemplate<
        true, DGCExecutionSetType::Shader_Dispatch, IDCMCore_NS::Float3>>;

    IDVC_NS::UniquePtr<IDVC_NS::ShaderObject> mComputeShader1;
    IDVC_NS::UniquePtr<IDVC_NS::ShaderObject> mComputeShader2;

    void prepare_compute_sequence_shader();
    vk::IndirectExecutionSetEXT _compute_shader_executionSet;
    uint32_t _preprocess_shader_Size;
    vk::Buffer _preprocess_shader_Buffer;
    vk::DeviceMemory _preprocess_shader_BufferMem;
    vk::DeviceAddress _preprocess_shader_BufAddr;

    void dgc_dispatch_shader(vk::CommandBuffer cmd);

    IDVC_NS::UniquePtr<DispatchSequence_Shader> mDispatchSequenceShader {};
};

VE_CREATE_APPLICATION(DGCTest, 1600, 900);