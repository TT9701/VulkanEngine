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

class Application {
public:
    Application(ApplicationSpecification const& spec);
    virtual ~Application();
    MOVABLE_ONLY(Application);

public:
    void Run();

public:
    ImmediateSubmitManager const* GetImmediateSubmitManager() const {
        return &mImmSubmitMgr;
    }

protected:
    UniquePtr<SDLWindow> CreateSDLWindow(const char* name, uint32_t width,
                                         uint32_t height);
    UniquePtr<Context> CreateContext();
    UniquePtr<Swapchain> CreateSwapchain();

    RenderResourceManager CreateRenderResourceManager();
    ImmediateSubmitManager CreateImmediateSubmitManager();
    CommandManager CreateCommandManager();
    PipelineManager CreatePipelineManager();
    ShaderManager CreateShaderManager();

    virtual void CreatePipelines();
    virtual void LoadShaders();
    virtual void PollEvents(SDL_Event* e, float deltaTime);
    virtual void Update_OnResize();
    virtual void UpdateScene();
    virtual void Prepare();

    virtual void BeginFrame();
    virtual void RenderFrame();
    virtual void EndFrame();

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage();

    void CreateExternalTriangleData();
    void CreateCUDASyncStructures();
    void SetCudaInterop();
#endif

protected:
    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow> mWindow;
    UniquePtr<Context> mContext;
    UniquePtr<Swapchain> mSwapchain;

    RenderResourceManager mRenderResMgr;
    CommandManager mCmdMgr;
    ImmediateSubmitManager mImmSubmitMgr;
    PipelineManager mPipelineMgr;
    ShaderManager mShaderMgr;

    IntelliDesign_NS::Core::Utils::FrameTimer mFrameTimer;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;

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
