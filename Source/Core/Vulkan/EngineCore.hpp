#pragma once

#include <vulkan/vulkan.hpp>
#include "CommandManager.hpp"
#include "Core/Model/Mesh.hpp"
#include "Core/Model/Model.hpp"
#include "Core/Utilities/Camera.hpp"
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "Pipeline.hpp"

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
class CommandBuffers;
class CommandPool;
class DescriptorManager;
class Shader;
class RenderResource;

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
    ImmediateSubmitManager* GetImmediateSubmitManager() const {
        return mSPImmediateSubmitManager.get();
    }

private:
    void Draw();

    UniquePtr<SDLWindow> CreateSDLWindow();
    UniquePtr<Context> CreateContext();
    UniquePtr<Swapchain> CreateSwapchain();
    SharedPtr<RenderResource> CreateDrawImage();
    SharedPtr<RenderResource> CreateDepthImage();
    UniquePtr<ImmediateSubmitManager> CreateImmediateSubmitManager();
    UniquePtr<CommandManager> CreateCommandManager();
    SharedPtr<RenderResource> CreateErrorCheckTexture();
    UniquePtr<DescriptorManager> CreateDescriptorManager();
    UniquePtr<PipelineManager> CreatePipelineManager();

    void CreatePipelines();
    void CreateDescriptors();

    void UpdateScene();
    void UpdateSceneUBO();

    // Compute
    void CreateBackgroundComputeDescriptors();
    void CreateBackgroundComputePipeline();

    // Graphics
    void CreateMeshPipeline();
    void CreateMeshDescriptors();

    // Draw quad
    void CreateDrawQuadDescriptors();
    void CreateDrawQuadPipeline();

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage();

    void CreateExternalTriangleData();
    void CreateCUDASyncStructures();
    void SetCudaInterop();
#endif

    void DrawBackground(vk::CommandBuffer cmd);
    void DrawMesh(vk::CommandBuffer cmd);
    void DrawQuad(vk::CommandBuffer cmd);

private:
    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow> mSPWindow;
    UniquePtr<Context> mSPContext;
    UniquePtr<Swapchain> mSPSwapchain;

    SharedPtr<RenderResource> mDrawImage;
    SharedPtr<RenderResource> mDepthImage;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
#endif
    UniquePtr<CommandManager> mSPCmdManager;
    UniquePtr<ImmediateSubmitManager> mSPImmediateSubmitManager;
    SharedPtr<RenderResource> mErrorCheckImage;
    UniquePtr<DescriptorManager> mDescriptorManager;
    UniquePtr<PipelineManager> mPipelineManager;

    SharedPtr<RenderResource> mSceneUniformBuffer {};

    SharedPtr<RenderResource> mRWBuffer {};

    Camera mMainCamera {};

    SceneData mSceneData {};

#ifdef CUDA_VULKAN_INTEROP
    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
#endif

    SharedPtr<Model> mFactoryModel;
};

}  // namespace IntelliDesign_NS::Vulkan::Core