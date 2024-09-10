#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Model/Mesh.hpp"
#include "Core/Model/Model.hpp"
#include "Core/Utilities/Camera.hpp"
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/Vulkan/Manager/CommandManager.hpp"

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
class RenderResourceManager;
class ShaderManager;
class PipelineManager;

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
        return mPImmediateSubmitManager.get();
    }

private:
    void Draw();

    UniquePtr<SDLWindow> CreateSDLWindow();
    UniquePtr<Context> CreateContext();
    UniquePtr<Swapchain> CreateSwapchain();
    UniquePtr<RenderResourceManager> CreateRenderResourceManager();
    UniquePtr<ImmediateSubmitManager> CreateImmediateSubmitManager();
    UniquePtr<CommandManager> CreateCommandManager();
    UniquePtr<PipelineManager> CreatePipelineManager();
    UniquePtr<ShaderManager> CreateShaderModuleManager();
    UniquePtr<DescriptorManager> CreateDescriptorBufferManager();
    void CreateDrawImage();
    void CreateDepthImage();
    void CreateErrorCheckTexture();
    void CreatePipelines();
    void CreateDescriptors();

    void UpdateScene();
    void UpdateSceneUBO();

    void LoadShaders();

    // Compute
    void CreateBackgroundComputeDescriptors();
    void CreateBackgroundComputePipeline();

    // Graphics
    void CreateMeshPipeline();
    void CreateMeshDescriptors();

    void CreateMeshShaderPipeline();
    void CreateMeshShaderDescriptors();

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
    void MeshShaderDraw(vk::CommandBuffer cmd);

private:
    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow> mPWindow;
    UniquePtr<Context> mPContext;
    UniquePtr<Swapchain> mPSwapchain;

    UniquePtr<RenderResourceManager> mRenderResManager;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
#endif
    UniquePtr<CommandManager> mPCmdManager;
    UniquePtr<ImmediateSubmitManager> mPImmediateSubmitManager;
    UniquePtr<DescriptorManager> mDescriptorManager;
    UniquePtr<PipelineManager> mPipelineManager;
    UniquePtr<ShaderManager> mShaderManager;

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