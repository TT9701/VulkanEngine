#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Model/Mesh.hpp"
#include "Core/Model/Model.hpp"
#include "Core/Utilities/Camera.hpp"
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/Vulkan/Manager/CommandManager.hpp"
#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.hpp"
#include "Core/Vulkan/Manager/RenderResourceManager.hpp"
#include "Core/Vulkan/Manager/ShaderManager.hpp"

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"
#endif

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

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

class EngineCore {
public:
    EngineCore();
    ~EngineCore();
    MOVABLE_ONLY(EngineCore);

public:
    void Run();

public:
    ImmediateSubmitManager const* GetImmediateSubmitManager() const {
        return &mImmSubmitMgr;
    }

private:
    void Draw();

    UniquePtr<SDLWindow> CreateSDLWindow();
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

    DrawCallManager mBackgroundDrawCallMgr;
    DrawCallManager mMeshDrawCallMgr;
    DrawCallManager mMeshShaderDrawCallMgr;
    DrawCallManager mQuadDrawCallMgr;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
#endif

    Camera mMainCamera {};
    SceneData mSceneData {};
    SharedPtr<Model> mFactoryModel {};

#ifdef CUDA_VULKAN_INTEROP
    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
#endif
};
}  // namespace IntelliDesign_NS::Vulkan::Core