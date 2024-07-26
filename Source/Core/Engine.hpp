#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanCommandManager.hpp"
#include "VulkanDescriptors.hpp"

#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"

class SDLWindow;
class VulkanContext;
class VulkanMemoryAllocator;
class VulkanExternalMemoryPool;
class VulkanSwapchain;
class VulkanAllocatedImage;
class VulkanFence;
class VulkanCommandBuffer;
class VulkanCommandPool;

struct FrameData {
    // vk::Semaphore mReady4RenderSemaphore {}, mReady4PresentSemaphore {};
    // vk::Fence mRenderFence {};

    vk::CommandPool mCommandPool {};
    vk::CommandBuffer mCommandBuffer {};
};

constexpr uint32_t FRAME_OVERLAP = 3;

class VulkanEngine {
    USING_TEMPLATE_PTR_TYPE(Type_PInstance, Type_SPInstance);

public:
    VulkanEngine();
    ~VulkanEngine();
    MOVABLE_ONLY(VulkanEngine);

public:
    void Init();
    void Run();

public:
    FrameData& GetCurrentFrameData() {
        return mFrameDatas[mFrameNum % FRAME_OVERLAP];
    }

    Type_SPInstance<ImmediateSubmitManager> GetImmediateSubmitManager() {
        return mSPImmediateSubmitManager;
    }

private:
    void Draw();

    void InitVulkan();

    Type_SPInstance<SDLWindow> CreateSDLWindow();

    Type_SPInstance<VulkanContext> CreateContext();

    Type_SPInstance<VulkanSwapchain> CreateSwapchain();

    Type_PInstance<VulkanAllocatedImage> CreateDrawImage();

    Type_PInstance<CUDA::VulkanExternalImage> CreateExternalImage();

    Type_SPInstance<ImmediateSubmitManager> CreateImmediateSubmitManager();

    void CreateCommands();

    void CreateSyncStructures();

    void CreatePipelines();

    void CreateDescriptors();

    void CreateTriangleData();
    void CreateExternalTriangleData();

    Type_PInstance<VulkanAllocatedImage> CreateErrorCheckTexture();
    void CreateDefaultSamplers();

    void SetCudaInterop();

    GPUMeshBuffers UploadMeshData(::std::span<uint32_t> indices,
                                  ::std::span<Vertex> vertices);

    // Compute
    void CreateBackgroundComputeDescriptors();
    void CreateBackgroundComputePipeline();

    // Graphics
    void CreateTrianglePipeline();
    void CreateTriangleDescriptors();

    void DrawBackground(vk::CommandBuffer cmd);
    void DrawTriangle(vk::CommandBuffer cmd);

private:
    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    Type_SPInstance<SDLWindow> mSPWindow;

    Type_SPInstance<VulkanContext> mSPContext;

    Type_SPInstance<VulkanSwapchain> mSPSwapchain;

    Type_PInstance<VulkanAllocatedImage> mDrawImage;

    Type_PInstance<CUDA::VulkanExternalImage> mCUDAExternalImage;

    Type_SPInstance<ImmediateSubmitManager> mSPImmediateSubmitManager;

    ::std::array<FrameData, FRAME_OVERLAP> mFrameDatas {};

    Type_SPInstance<VulkanAllocatedImage> mErrorCheckImage;

    vk::DescriptorSet mDrawImageDescriptors {};
    vk::DescriptorSetLayout mDrawImageDescriptorLayout {};

    DescriptorAllocator mMainDescriptorAllocator {};

    // background compute
    vk::Pipeline mBackgroundComputePipeline {};
    vk::PipelineLayout mBackgroundComputePipelineLayout {};

    // graphic pipeline
    vk::Pipeline mTrianglePipelie {};
    vk::PipelineLayout mTrianglePipelieLayout {};
    GPUMeshBuffers mTriangleMesh {};

    ExternalGPUMeshBuffers mTriangleExternalMesh {};

    vk::DescriptorSetLayout mTextureTriangleDescriptorLayout {};
    vk::DescriptorSet mTextureTriangleDescriptors {};

    vk::Sampler mDefaultSamplerLinear;
    vk::Sampler mDefaultSamplerNearest;

    CUDA::VulkanExternalSemaphore mCUDAWaitSemaphore {};
    CUDA::VulkanExternalSemaphore mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};
};