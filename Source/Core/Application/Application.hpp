#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Core/Model/CISDI_3DModelConverter.hpp"
#include "Core/Model/Model.hpp"
#include "Core/Platform/Window.hpp"
#include "Core/Utilities/Camera.hpp"
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/Timer.hpp"
#include "Core/Vulkan/Manager/CommandManager.hpp"
#include "Core/Vulkan/Manager/Context.hpp"
#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Core/Vulkan/Manager/PipelineManager.hpp"
#include "Core/Vulkan/Manager/RenderResourceManager.hpp"
#include "Core/Vulkan/Manager/ShaderManager.hpp"
#include "Core/Vulkan/Native/Swapchain.hpp"
#include "Core/Vulkan/RenderPass/RenderPassBindingInfo.hpp"

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"
#endif

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

struct ApplicationCommandLineArgs {
    int Count = 0;
    char** Args = nullptr;

    const char* operator[](int index) const {
        VE_ASSERT(index < Count, "");
        return Args[index];
    }
};

struct ApplicationSpecification {
    ::std::string Name = "Vulkan Application";
    uint32_t width, height;
    ApplicationCommandLineArgs CommandLineArgs;
};

class Context;
class MemoryAllocator;
class ExternalMemoryPool;
class Swapchain;
class Fence;

constexpr uint32_t FRAME_OVERLAP = 3;

struct SceneData {
    glm::vec4 sunLightPos {-2.0f, 3.0f, 1.0f, 1.0f};
    glm::vec4 sunLightColor {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 cameraPos {};
    glm::mat4 view {};
    glm::mat4 proj {};
    glm::mat4 viewProj {};
};

class Application {
public:
    Application(ApplicationSpecification const& spec);
    ~Application();
    MOVABLE_ONLY(Application);

public:
    void Run();

public:
    ImmediateSubmitManager const* GetImmediateSubmitManager() const {
        return &mImmSubmitMgr;
    }

private:
    void Draw();

    UniquePtr<SDLWindow> CreateSDLWindow(const char* name, uint32_t width,
                                         uint32_t height);
    UniquePtr<Context> CreateContext();
    UniquePtr<Swapchain> CreateSwapchain();

    RenderResourceManager CreateRenderResourceManager();
    ImmediateSubmitManager CreateImmediateSubmitManager();
    CommandManager CreateCommandManager();
    PipelineManager CreatePipelineManager();
    ShaderManager CreateShaderManager();
    DescriptorManager CreateDescriptorManager();

    void CreateDrawImage();
    void CreateDepthImage();
    void CreateErrorCheckTexture();
    void CreatePipelines();

    void UpdateScene();
    void UpdateSceneUBO();

    void LoadShaders();

    void CreateBackgroundComputePipeline();
    void CreateMeshPipeline();
    void CreateMeshShaderPipeline();
    void CreateDrawQuadPipeline();

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage();

    void CreateExternalTriangleData();
    void CreateCUDASyncStructures();
    void SetCudaInterop();
#endif

    void RecordDrawBackgroundCmds();
    void RecordDrawMeshCmds();
    void RecordDrawQuadCmds();
    void RecordMeshShaderDrawCmds();

private:
    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow> mWindow;
    UniquePtr<Context> mContext;
    UniquePtr<Swapchain> mSwapchain;

    DescriptorManager mDescMgr;
    RenderResourceManager mRenderResMgr;
    CommandManager mCmdMgr;
    ImmediateSubmitManager mImmSubmitMgr;
    PipelineManager mPipelineMgr;
    ShaderManager mShaderMgr;

    RenderPassBindingInfo mMeshShaderPass;

    DrawCallManager mBackgroundDrawCallMgr;
    DrawCallManager mMeshDrawCallMgr;
    // DrawCallManager mMeshShaderDrawCallMgr;
    DrawCallManager mQuadDrawCallMgr;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
#endif

    Camera mMainCamera {};
    SceneData mSceneData {};
    SharedPtr<Model> mFactoryModel {};
    IntelliDesign_NS::Core::Utils::FrameTimer mFrameTimer;

#ifdef CUDA_VULKAN_INTEROP
    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
#endif
};

// To be defined in CLIENT
Application* CreateApplication(ApplicationCommandLineArgs args);

}  // namespace IntelliDesign_NS::Vulkan::Core

#define VE_CREATE_APPLICATION(CLASS_NAME, WIDTH, HEIGHT)                \
    IntelliDesign_NS::Vulkan::Core::Application*                        \
    IntelliDesign_NS::Vulkan::Core::CreateApplication(                  \
        ApplicationCommandLineArgs args) {                              \
        IntelliDesign_NS::Vulkan::Core::ApplicationSpecification spec { \
            VE_STRINGIFY_MACRO(CLASS_NAME), WIDTH, HEIGHT, args};       \
        return new CLASS_NAME {spec};                                   \
    }
