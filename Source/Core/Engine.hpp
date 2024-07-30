#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanCommandManager.hpp"

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
class VulkanSampler;
class VulkanDescriptorManager;

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

    SharedPtr<SDLWindow>                 CreateSDLWindow();
    SharedPtr<VulkanContext>             CreateContext();
    SharedPtr<VulkanSwapchain>           CreateSwapchain();
    UniquePtr<VulkanAllocatedImage>      CreateDrawImage();
    UniquePtr<CUDA::VulkanExternalImage> CreateExternalImage();
    SharedPtr<ImmediateSubmitManager>    CreateImmediateSubmitManager();
    SharedPtr<VulkanCommandManager>      CreateCommandManager();
    UniquePtr<VulkanAllocatedImage>      CreateErrorCheckTexture();
    SharedPtr<VulkanDescriptorManager>   CreateDescriptorManager();

    void CreateCUDASyncStructures();
    void CreatePipelines();
    void CreateDescriptors();
    void CreateTriangleData();
    void CreateExternalTriangleData();
    void CreateDefaultSamplers();
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

    SharedPtr<SDLWindow>                 mSPWindow;
    SharedPtr<VulkanContext>             mSPContext;
    SharedPtr<VulkanSwapchain>           mSPSwapchain;
    UniquePtr<VulkanAllocatedImage>      mDrawImage;
    UniquePtr<CUDA::VulkanExternalImage> mCUDAExternalImage;
    SharedPtr<VulkanCommandManager>      mSPCmdManager;
    SharedPtr<ImmediateSubmitManager>    mSPImmediateSubmitManager;
    SharedPtr<VulkanAllocatedImage>      mErrorCheckImage;

    SharedPtr<VulkanDescriptorManager> mDescriptorManager;

    // background compute
    vk::Pipeline       mBackgroundComputePipeline {};
    vk::PipelineLayout mBackgroundComputePipelineLayout {};

    // graphic pipeline
    vk::Pipeline       mTrianglePipelie {};
    vk::PipelineLayout mTrianglePipelieLayout {};
    GPUMeshBuffers     mTriangleMesh {};

    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    SharedPtr<VulkanSampler> mDefaultSamplerLinear {};
    SharedPtr<VulkanSampler> mDefaultSamplerNearest {};

    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDAWaitSemaphore {};
    SharedPtr<CUDA::VulkanExternalSemaphore> mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
};