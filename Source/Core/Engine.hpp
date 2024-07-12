#pragma once

#include <vulkan/vulkan_raii.hpp>
#include "Mesh.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanDescriptors.hpp"
#include "VulkanImage.hpp"

#include "CUDA/CUDAStream.h"
#include "CUDA/CUDAVulkan.h"

struct SDL_Window;

struct FrameData {
    vk::Semaphore mReady4RenderSemaphore {}, mReady4PresentSemaphore {};
    vk::Fence mRenderFence {};

    vk::CommandPool mCommandPool {};
    vk::CommandBuffer mCommandBuffer {};
};

constexpr uint32_t FRAME_OVERLAP = 2;

constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

class VulkanEngine {
public:
    template <class T>
    using Type_PInstance =
        IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<T>;

public:
    VulkanEngine();
    ~VulkanEngine();

    void Init();
    void Run();

public:
    FrameData& GetCurrentFrameData() {
        return mFrameDatas[mFrameNum % FRAME_OVERLAP];
    }

    VmaAllocator const& GetVmaAllocator() const { return *mVmaAllocator; }

    vk::Device const& GetVkDevice() const { return *mDevice; }

    void ImmediateSubmit(
        ::std::function<void(vk::CommandBuffer cmd)>&& function);

private:
    void Draw();

    void InitVulkan();

    ::std::pmr::memory_resource* CreateGlobalMemoryPool();

    SDL_Window* CreateSDLWindow();

    Type_PInstance<vk::Instance> CreateInstance();
#ifdef DEBUG
    Type_PInstance<vk::DebugUtilsMessengerEXT> CreateDebugUtilsMessenger();
#endif
    Type_PInstance<VkSurfaceKHR> CreateSurface();

    vk::PhysicalDevice PickPhysicalDevice();
    void SetQueueFamily(vk::PhysicalDevice physicalDevice,
                        vk::QueueFlags requestedQueueTypes);

    Type_PInstance<vk::Device> CreateDevice();

    Type_PInstance<VmaAllocator> CreateVmaAllocators();
    Type_PInstance<VmaPool> CreateVmaExternalMemoryPool();

    Type_PInstance<vk::SwapchainKHR> CreateSwapchain();
    Type_PInstance<AllocatedVulkanImage> CreateDrawImage();
    Type_PInstance<CUDA::VulkanExternalImage> CreateExternalImage();

    void CreateCommands();

    void CreateSyncStructures();

    void CreatePipelines();

    void CreateDescriptors();

    void CreateTriangleData();
    void CreateExternalTriangleData();

    void CreateErrorCheckTextures();
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
    // helper functions
    void SetInstanceLayers(
        ::std::vector<::std::string> const& requestedLayers = {});
    void SetInstanceExtensions(
        ::std::vector<::std::string> const& requestedExtensions = {});

    std::vector<std::string> GetSDLRequestedInstanceExtensions() const;

private:
    ::std::pmr::memory_resource* mPMemPool {nullptr};

    bool mStopRendering {false};
    uint32_t mFrameNum {0};

    int mWindowWidth {1600};
    int mWindowHeight {900};
    SDL_Window* mWindow {nullptr};

    ::std::vector<::std::string> mEnabledInstanceLayers {};
    ::std::vector<::std::string> mEnabledInstanceExtensions {};

    Type_PInstance<vk::Instance> mPInstance {nullptr};
#ifdef DEBUG
    Type_PInstance<vk::DebugUtilsMessengerEXT> mPDebugUtilsMessenger {nullptr};
#endif
    Type_PInstance<VkSurfaceKHR> mPSurface {nullptr};

    std::optional<uint32_t> mGraphicsFamilyIndex;
    uint32_t mGraphicsQueueCount = 0;
    std::optional<uint32_t> mComputeFamilyIndex;
    uint32_t mComputeQueueCount = 0;
    std::optional<uint32_t> mTransferFamilyIndex;
    uint32_t mTransferQueueCount = 0;

    vk::PhysicalDevice mPhysicalDevice {};

    ::std::vector<vk::Queue> mGraphicQueues {};
    ::std::vector<vk::Queue> mComputeQueues {};
    ::std::vector<vk::Queue> mTransferQueues {};

    Type_PInstance<vk::Device> mDevice {nullptr};

    Type_PInstance<VmaAllocator> mVmaAllocator {nullptr};

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};
    Type_PInstance<VmaPool> mVmaExternalMemoryPool {nullptr};

    vk::Format mSwapchainImageFormat {};
    ::std::vector<vk::Image> mSwapchainImages {};
    ::std::vector<vk::ImageView> mSwapchainImageViews {};
    vk::Extent2D mSwapchainExtent {};

    Type_PInstance<AllocatedVulkanImage> mDrawImage {nullptr};

    Type_PInstance<CUDA::VulkanExternalImage> mCUDAExternalImage {nullptr};

    Type_PInstance<vk::SwapchainKHR> mSwapchain {nullptr};

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

    AllocatedVulkanImage mErrorCheckImage {};

    vk::Sampler mDefaultSamplerLinear;
    vk::Sampler mDefaultSamplerNearest;

    CUDA::VulkanExternalSemaphore mCUDAWaitSemaphore {};
    CUDA::VulkanExternalSemaphore mCUDASignalSemaphore {};

    CUDA::CUDAStream mCUDAStream {};

    ::std::array<FrameData, FRAME_OVERLAP> mFrameDatas {};

    struct ImmediateSubmit {
        vk::Fence mFence {};
        vk::CommandBuffer mCommandBuffer {};
        vk::CommandPool mCommandPool {};
    } mImmediateSubmit;
};