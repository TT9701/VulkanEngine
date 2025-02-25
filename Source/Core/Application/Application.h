#pragma once

#include "Core/Model/GPUGeometryData.h"
#include "Core/Platform/Window.h"
#include "Core/Utilities/Camera.h"
#include "Core/Utilities/Defines.h"
#include "Core/Utilities/GUI.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Utilities/Timer.h"
#include "Core/Vulkan/Manager/CommandManager.h"
#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/RenderFrame.h"
#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Manager/ShaderManager.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Core/Vulkan/Native/Swapchain.h"
#include "Core/Vulkan/RenderGraph/RenderPassBindingInfo.h"
#include "Core/Vulkan/RenderGraph/RenderSequenceConfig.h"

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

class VulkanContext;
class MemoryAllocator;
class ExternalMemoryPool;
class Swapchain;
class Fence;

constexpr uint32_t FRAME_OVERLAP = 3;

class Application {
public:
    Application(ApplicationSpecification const& spec);
    virtual ~Application();
    CLASS_NO_COPY_MOVE(Application);

public:
    void Run();

protected:
    virtual void CreatePipelines();
    virtual void LoadShaders();
    virtual void PollEvents(SDL_Event* e, float deltaTime);
    virtual void Update_OnResize();
    virtual void UpdateScene();
    virtual void Prepare();

    virtual void BeginFrame(Core::RenderFrame& frame);
    virtual void RenderFrame(Core::RenderFrame& frame);
    virtual void EndFrame(Core::RenderFrame& frame);

    virtual void RenderToSwapchainBindings(vk::CommandBuffer cmd) = 0;

    SDLWindow& GetSDLWindow() const;
    VulkanContext& GetVulkanContext() const;
    Swapchain& GetSwapchain() const;
    RenderResourceManager& GetRenderResMgr() const;
    CommandManager& GetCmdMgr() const;
    PipelineManager& GetPipelineMgr() const;
    ShaderManager& GetShaderMgr() const;
    GUI& GetUILayer();

    Core::RenderFrame& GetCurFrame();
    Type_STLVector<Core::RenderFrame> const& GetFrames() const;
    Type_STLVector<Core::RenderFrame>& GetFrames();
    
    bool mStopRendering {false};
    uint64_t mFrameNum {0};
    IntelliDesign_NS::Core::Utils::FrameTimer mFrameTimer;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage();

    void CreateExternalTriangleData();
    void CreateCUDASyncStructures();
    void SetCudaInterop();

    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;

    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
#endif

private:
    UniquePtr<SDLWindow> CreateSDLWindow(const char* name, uint32_t width,
                                         uint32_t height);
    UniquePtr<VulkanContext> CreateContext();
    UniquePtr<Swapchain> CreateSwapchain();

    UniquePtr<RenderResourceManager> CreateRenderResourceManager();
    UniquePtr<CommandManager> CreateCommandManager();
    UniquePtr<PipelineManager> CreatePipelineManager();
    UniquePtr<ShaderManager> CreateShaderManager();

    UniquePtr<SDLWindow> mWindow;
    UniquePtr<VulkanContext> mVulkanContext;
    UniquePtr<Swapchain> mSwapchain;
    UniquePtr<RenderResourceManager> mRenderResMgr;
    UniquePtr<CommandManager> mCmdMgr;
    UniquePtr<PipelineManager> mPipelineMgr;
    UniquePtr<ShaderManager> mShaderMgr;

    GUI mGui;

    Type_STLVector<Core::RenderFrame> mFrames;
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
