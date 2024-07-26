#pragma once

#include <vulkan/vulkan.hpp>
#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
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

class ImmediateSubmitManager {
    USING_TEMPLATE_PTR_TYPE(Type_PInstance, Type_SPInstance);

public:
    ImmediateSubmitManager(Type_SPInstance<VulkanContext> const& ctx,
                           uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    Type_SPInstance<VulkanFence> CreateFence();
    Type_SPInstance<VulkanCommandBuffer> CreateCommandBuffer();
    Type_SPInstance<VulkanCommandPool> CreateCommandPool();

private:
    Type_SPInstance<VulkanContext> pContex;
    uint32_t mQueueFamilyIndex;

    Type_SPInstance<VulkanFence> mSPFence;
    Type_SPInstance<VulkanCommandPool> mSPCommandPool;
    Type_SPInstance<VulkanCommandBuffer> mSPCommandBuffer;
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
        return mSPImmediateSubmit;
    }

private:
    void Draw();

    void InitVulkan();

    Type_SPInstance<SDLWindow> CreateSDLWindow();

    Type_SPInstance<VulkanContext> CreateContext();

    Type_SPInstance<VulkanMemoryAllocator> CreateVmaAllocator();
    Type_SPInstance<VulkanExternalMemoryPool> CreateVmaExternalMemoryPool();

    Type_SPInstance<VulkanSwapchain> CreateSwapchain();

    Type_PInstance<VulkanAllocatedImage> CreateDrawImage();
    Type_PInstance<CUDA::VulkanExternalImage> CreateExternalImage();

    Type_SPInstance<ImmediateSubmitManager> CreateImmediateSubmit();

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

    Type_SPInstance<SDLWindow> mSPWindow {nullptr};

    Type_SPInstance<VulkanContext> mSPContext {nullptr};

    Type_SPInstance<VulkanMemoryAllocator> mSPVmaAllocator {nullptr};

    Type_SPInstance<VulkanExternalMemoryPool> mVmaExternalMemoryPool {nullptr};

    Type_PInstance<VulkanAllocatedImage> mDrawImage {nullptr};

    Type_PInstance<CUDA::VulkanExternalImage> mCUDAExternalImage {nullptr};

    Type_SPInstance<VulkanSwapchain> mSPSwapchain {nullptr};

    Type_SPInstance<ImmediateSubmitManager> mSPImmediateSubmit;

    ::std::array<FrameData, FRAME_OVERLAP> mFrameDatas {};

    Type_SPInstance<VulkanAllocatedImage> mErrorCheckImage {nullptr};

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