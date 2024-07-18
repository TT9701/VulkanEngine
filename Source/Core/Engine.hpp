#pragma once

#include <vulkan/vulkan.hpp>
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanDescriptors.hpp"
#include "VulkanHelper.hpp"

#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"

class SDLWindow;
class VulkanInstance;
class VulkanSurface;
class VulkanDebugUtils;
class VulkanPhysicalDevice;
class VulkanDevice;
class VulkanMemoryAllocator;
class VulkanExternalMemoryPool;
class VulkanSwapchain;
class VulkanAllocatedImage;

struct FrameData {
    vk::Semaphore mReady4RenderSemaphore {}, mReady4PresentSemaphore {};
    vk::Fence mRenderFence {};

    vk::CommandPool mCommandPool {};
    vk::CommandBuffer mCommandBuffer {};
};

constexpr uint32_t FRAME_OVERLAP = 2;

constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

class VulkanEngine {
    USING_TEMPLATE_PTR_TYPE(Type_PInstance, Type_SPInstance);

public:
    VulkanEngine();
    ~VulkanEngine();

    void Init();
    void Run();

public:
    FrameData& GetCurrentFrameData() {
        return mFrameDatas[mFrameNum % FRAME_OVERLAP];
    }

    void ImmediateSubmit(
        ::std::function<void(vk::CommandBuffer cmd)>&& function);

private:
    void Draw();

    void InitVulkan();

    ::std::pmr::memory_resource* CreateGlobalMemoryPool();

    Type_SPInstance<SDLWindow> CreateSDLWindow();

    Type_SPInstance<VulkanInstance> CreateInstance();
#ifdef DEBUG
    Type_PInstance<VulkanDebugUtils> CreateDebugUtilsMessenger();
#endif
    Type_SPInstance<VulkanSurface> CreateSurface();

    Type_SPInstance<VulkanPhysicalDevice> PickPhysicalDevice();

    Type_SPInstance<VulkanDevice> CreateDevice();

    Type_SPInstance<VulkanMemoryAllocator> CreateVmaAllocator();
    Type_SPInstance<VulkanExternalMemoryPool> CreateVmaExternalMemoryPool();

    Type_SPInstance<VulkanSwapchain> CreateSwapchain();

    Type_PInstance<VulkanAllocatedImage> CreateDrawImage();
    Type_PInstance<CUDA::VulkanExternalImage> CreateExternalImage();

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
    ::std::pmr::memory_resource* mPMemPool {nullptr};

    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    Type_SPInstance<SDLWindow> mSPWindow {nullptr};

    Type_SPInstance<VulkanInstance> mSPInstance {nullptr};
#ifdef DEBUG
    Type_PInstance<VulkanDebugUtils> mPDebugUtilsMessenger {nullptr};
#endif
    Type_SPInstance<VulkanSurface> mSPSurface {nullptr};

    Type_SPInstance<VulkanPhysicalDevice> mSPPhysicalDevice {nullptr};

    Type_SPInstance<VulkanDevice> mSPDevice {nullptr};

    Type_SPInstance<VulkanMemoryAllocator> mSPVmaAllocator {nullptr};

    Type_SPInstance<VulkanExternalMemoryPool> mVmaExternalMemoryPool {nullptr};

    Type_PInstance<VulkanAllocatedImage> mDrawImage {nullptr};

    Type_PInstance<CUDA::VulkanExternalImage> mCUDAExternalImage {nullptr};

    Type_SPInstance<VulkanSwapchain> mSwapchain {nullptr};

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

    struct ImmediateSubmit {
        vk::Fence mFence {};
        vk::CommandBuffer mCommandBuffer {};
        vk::CommandPool mCommandPool {};
    } mImmediateSubmit;
};