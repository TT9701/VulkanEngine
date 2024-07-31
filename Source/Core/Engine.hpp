#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanCommandManager.hpp"
#include "VulkanPipeline.hpp"

#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"

class SDLWindow;
class VulkanContext;
class VulkanMemoryAllocator;
class VulkanExternalMemoryPool;
class VulkanSwapchain;
class VulkanAllocatedImage;
class VulkanFence;
class VulkanCommandBuffers;
class VulkanCommandPool;
class VulkanDescriptorManager;
class VulkanShader;

constexpr uint32_t FRAME_OVERLAP = 3;

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

    UniquePtr<SDLWindow>                 CreateSDLWindow();
    UniquePtr<VulkanContext>             CreateContext();
    UniquePtr<VulkanSwapchain>           CreateSwapchain();
    SharedPtr<VulkanAllocatedImage>      CreateDrawImage();
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage();
    UniquePtr<ImmediateSubmitManager>    CreateImmediateSubmitManager();
    UniquePtr<VulkanCommandManager>      CreateCommandManager();
    SharedPtr<VulkanAllocatedImage>      CreateErrorCheckTexture();
    UniquePtr<VulkanDescriptorManager>   CreateDescriptorManager();
    UniquePtr<VulkanPipelineManager>     CreatePipelineManager();

    void CreateCUDASyncStructures();
    void CreatePipelines();
    void CreateDescriptors();
    void CreateTriangleData();
    void CreateExternalTriangleData();
    void SetCudaInterop();

    GPUMeshBuffers UploadMeshData(::std::span<uint32_t> indices,
                                  ::std::span<Vertex>   vertices);

    // Compute
    void CreateBackgroundComputeDescriptors();
    void CreateBackgroundComputePipeline();

    // Graphics
    void CreateTrianglePipeline();
    void CreateTriangleDescriptors();

    void DrawBackground(vk::CommandBuffer cmd);
    void DrawTriangle(vk::CommandBuffer cmd);

private:
    bool     mStopRendering {false};
    uint32_t mFrameNum {0};

    UniquePtr<SDLWindow>                 mSPWindow;
    UniquePtr<VulkanContext>             mSPContext;
    UniquePtr<VulkanSwapchain>           mSPSwapchain;
    SharedPtr<VulkanAllocatedImage>      mDrawImage;
    SharedPtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
    UniquePtr<VulkanCommandManager>      mSPCmdManager;
    UniquePtr<ImmediateSubmitManager>    mSPImmediateSubmitManager;
    SharedPtr<VulkanAllocatedImage>      mErrorCheckImage;
    UniquePtr<VulkanDescriptorManager>   mDescriptorManager;
    UniquePtr<VulkanPipelineManager>     mPipelineManager;

    GPUMeshBuffers mTriangleMesh {};

    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
};