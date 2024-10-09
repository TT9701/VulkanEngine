#pragma once

#include "Core/Application/Application.hpp"
#include "Core/Application/EntryPoint.h"

namespace IDNS_VC = IntelliDesign_NS::Vulkan::Core;
namespace IDNC_CMP = IntelliDesign_NS::Core::MemoryPool;

struct SceneData {
    glm::vec4 sunLightPos {-2.0f, 3.0f, 1.0f, 1.0f};
    glm::vec4 sunLightColor {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 cameraPos {};
    glm::mat4 view {};
    glm::mat4 proj {};
    glm::mat4 viewProj {};
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
    void CreateErrorCheckTexture();

    void CreateBackgroundComputePipeline();
    void CreateMeshPipeline();
    void CreateMeshShaderPipeline();
    void CreateDrawQuadPipeline();

    void RecordDrawBackgroundCmds();
    void RecordDrawMeshCmds();
    void RecordDrawQuadCmds();
    void RecordMeshShaderDrawCmds();

    void UpdateSceneUBO();

private:
    IDNS_VC::RenderPassBindingInfo mMeshShaderPass;
    IDNS_VC::DrawCallManager mBackgroundDrawCallMgr;
    IDNS_VC::DrawCallManager mMeshDrawCallMgr;
    // IDNS_VC::DrawCallManager mMeshShaderDrawCallMgr;
    IDNS_VC::DrawCallManager mQuadDrawCallMgr;

    Camera mMainCamera {};
    SceneData mSceneData {};
    IDNS_VC::SharedPtr<IDNS_VC::Model> mFactoryModel {};
};

VE_CREATE_APPLICATION(MeshShaderDemo, 1600, 900);