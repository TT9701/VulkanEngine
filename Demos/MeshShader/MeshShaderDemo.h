#pragma once

#include "Core/Application/Application.h"
#include "Core/Application/EntryPoint.h"

#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

#include "Core/Utilities/GUI.h"

namespace IDNS_VC = IntelliDesign_NS::Vulkan::Core;
namespace IDNC_CMP = IntelliDesign_NS::Core::MemoryPool;

struct SceneData {
    glm::vec4 sunLightPos {-0.4f, 0.6f, 0.2f, 1.0f};
    glm::vec4 sunLightColor {5.0f, 5.0f, 5.0f, 1.0f};
    glm::vec4 cameraPos {};
    glm::mat4 view {};
    glm::mat4 proj {};
    glm::mat4 viewProj {};

    glm::vec4 objColor {0.7f};
    glm::vec4 metallicRoughness {0.5f};
    int32_t texIndex {0};
};

class MeshShaderDemo : public IDNS_VC::Application {
public:
    MeshShaderDemo(IDNS_VC::ApplicationSpecification const& spec);

    ~MeshShaderDemo() override;

private:
    virtual void CreatePipelines() override;
    virtual void LoadShaders() override;
    virtual void PollEvents(SDL_Event* e, float deltaTime) override;
    virtual void Update_OnResize() override;
    virtual void UpdateScene() override;
    virtual void Prepare() override;

    virtual void BeginFrame() override;
    virtual void RenderFrame() override;
    virtual void EndFrame() override;

private:
    void CreateDrawImage();
    void CreateDepthImage();
    void CreateRandomTexture();

    void CreateBackgroundComputePipeline();
    void CreateMeshPipeline();
    void CreateMeshShaderPipeline();
    void CreateDrawQuadPipeline();

    void RecordDrawBackgroundCmds();
    void RecordDrawMeshCmds();
    void RecordDrawQuadCmds();
    void RecordMeshShaderDrawCmds();

    void UpdateSceneUBO();

    void PrepareUIContext();

private:
    IDNS_VC::DescriptorSetPool mDescSetPool;
    IDNS_VC::DescriptorSetPool mBindlessDescSetPool;

    IDNS_VC::SharedPtr<IDNS_VC::DescriptorSet> mBindlessSet;

    IDNS_VC::RenderPassBindingInfo_Copy mPrepassCopy;

    IDNS_VC::RenderPassBindingInfo_PSO mBackgroundPass_PSO;
    IDNS_VC::RenderPassBindingInfo_Barrier mBackgroundPass_Barrier;

    IDNS_VC::RenderPassBindingInfo_PSO mMeshDrawPass;
    IDNS_VC::RenderPassBindingInfo_Barrier mMeshDrawPass_Barrier;

    IDNS_VC::RenderPassBindingInfo_PSO mMeshShaderPass;
    IDNS_VC::RenderPassBindingInfo_Barrier mMeshShaderPass_Barrier;

    IDNS_VC::RenderPassBindingInfo_PSO mQuadDrawPass_PSO;
    IDNS_VC::RenderPassBindingInfo_Barrier mQuadDrawPass_Barrier_Pre;
    IDNS_VC::RenderPassBindingInfo_Barrier mQuadDrawPass_Barrier_Post;

    IDNS_VC::GUI mGui;

    Camera mMainCamera {};
    SceneData mSceneData {};
    IDNS_VC::SharedPtr<IDNS_VC::Geometry> mFactoryModel {};

    // IDNS_VC::Type_STLVector<IDNS_VC::SharedPtr<IDNS_VC::Geometry>> mModels {};
};

VE_CREATE_APPLICATION(MeshShaderDemo, 1600, 900);