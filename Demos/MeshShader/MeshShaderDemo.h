#pragma once

#include "Core/Application/Application.h"
#include "Core/Application/EntryPoint.h"

#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

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

    virtual void BeginFrame(IDNS_VC::RenderFrame& frame) override;
    virtual void RenderFrame(IDNS_VC::RenderFrame& frame) override;
    virtual void EndFrame(IDNS_VC::RenderFrame& frame) override;

    virtual void RenderToSwapchainBindings(vk::CommandBuffer cmd) override;

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
    IDNS_VC::RenderPassBindingInfo_Barrier CreateBindingInfo_Barrier(
        bool swapchain = false);
    IDNS_VC::RenderPassBindingInfo_Copy CreateBindingInfo_Copy();

private:
    IDNS_VC::DescriptorSetPool mDescSetPool;

    IDNS_VC::RenderSequence mRenderSequence;

    IDNS_VC::RenderPassBindingInfo_Copy mPrepassCopy;

    IDNS_VC::Semaphore mCopySem;
    IDNS_VC::Semaphore mCmpSem;

    Camera mMainCamera {};
    SceneData mSceneData {};
    IDNS_VC::SharedPtr<IDNS_VC::Geometry> mFactoryModel {};

    // IDNS_VC::Type_STLVector<IDNS_VC::SharedPtr<IDNS_VC::Geometry>> mModels {};

    IDNS_VC::Type_STLString mImageName0;
    IDNS_VC::Type_STLString mImageName1;
};

VE_CREATE_APPLICATION(MeshShaderDemo, 1600, 900);