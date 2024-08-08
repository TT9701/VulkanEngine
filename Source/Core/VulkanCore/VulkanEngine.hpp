#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Model/Mesh.hpp"
#include "Core/Model/Model.hpp"
#include "Core/Utilities/Camera.hpp"
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "VulkanCommandManager.hpp"
#include "VulkanPipeline.hpp"

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"
#endif

class SDLWindow;
class VulkanContext;
class VulkanMemoryAllocator;
class VulkanExternalMemoryPool;
class VulkanSwapchain;
class VulkanImage;
class VulkanFence;
class VulkanCommandBuffers;
class VulkanCommandPool;
class VulkanDescriptorManager;
class VulkanShader;

constexpr uint32_t FRAME_OVERLAP = 3;

struct SceneData {
    glm::vec4 sunLightPos {-2.0f, 3.0f, 1.0f, 1.0f};
    glm::vec4 sunLightColor {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 cameraPos {};
    glm::mat4 view {};
    glm::mat4 proj {};
    glm::mat4 viewProj {};
};

class VulkanEngine {
public:
    VulkanEngine();
    ~VulkanEngine();
    MOVABLE_ONLY(VulkanEngine);

public:
    void Run();

public:
    ImmediateSubmitManager* GetImmediateSubmitManager() const {
        return mSPImmediateSubmitManager.get();
    }

private:
    void Draw();

    UniquePtr<SDLWindow>               CreateSDLWindow();
    UniquePtr<VulkanContext>           CreateContext();
    UniquePtr<VulkanSwapchain>         CreateSwapchain();
    SharedPtr<VulkanImage>             CreateDrawImage();
    SharedPtr<VulkanImage>             CreateDepthImage();
    UniquePtr<ImmediateSubmitManager>  CreateImmediateSubmitManager();
    UniquePtr<VulkanCommandManager>    CreateCommandManager();
    SharedPtr<VulkanImage>             CreateErrorCheckTexture();
    UniquePtr<VulkanDescriptorManager> CreateDescriptorManager();
    UniquePtr<VulkanPipelineManager>   CreatePipelineManager();
    SharedPtr<VulkanBuffer>            CreateSceneUniformBuffer();
    SharedPtr<VulkanBuffer>            CreateRWBuffer();

    void CreatePipelines();
    void CreateDescriptors();
    void CreateBoxData();

    GPUMeshBuffers UploadMeshData(::std::span<uint32_t> indices,
                                  ::std::span<Vertex>   vertices);

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
    bool     mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow>       mSPWindow;
    UniquePtr<VulkanContext>   mSPContext;
    UniquePtr<VulkanSwapchain> mSPSwapchain;

    SharedPtr<VulkanImage> mDrawImage;
    SharedPtr<VulkanImage> mDepthImage;

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
#endif
    UniquePtr<VulkanCommandManager>    mSPCmdManager;
    UniquePtr<ImmediateSubmitManager>  mSPImmediateSubmitManager;
    SharedPtr<VulkanImage>             mErrorCheckImage;
    UniquePtr<VulkanDescriptorManager> mDescriptorManager;
    UniquePtr<VulkanPipelineManager>   mPipelineManager;

    SharedPtr<VulkanBuffer> mSceneUniformBuffer;

    SharedPtr<VulkanBuffer> mRWBuffer;

    Camera mMainCamera {};

    GPUMeshBuffers mBoxMesh {};

    SceneData mSceneData {};

#ifdef CUDA_VULKAN_INTEROP
    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
#endif

    SharedPtr<Model> mFactoryModel;
};