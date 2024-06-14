#pragma once

#include <vulkan/vulkan.hpp>

#include "Utilities/VulkanUtilities.hpp"
#include "VulkanImage.hpp"
#include "VulkanDescriptors.hpp"

struct SDL_Window;

struct FrameData {
    vk::Semaphore mReady4RenderSemaphore {}, mReady4PresentSemaphore {};
    vk::Fence     mRenderFence {};

    vk::CommandPool   mCommandPool {};
    vk::CommandBuffer mCommandBuffer {};

    Utils::DeletionQueue mDeletionQueue {};
};

constexpr uint32_t FRAME_OVERLAP = 2;

constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

class VulkanEngine {
public:
    void Init();
    void Run();
    void Cleanup();

public:
    ::std::array<FrameData, FRAME_OVERLAP> mFrameDatas;

    FrameData& GetCurrentFrameData() {
        return mFrameDatas[mFrameNum % FRAME_OVERLAP];
    }

private:
    void Draw();

    void InitSDLWindow();
    void InitVulkan();

    void CreateInstance();
#ifdef DEBUG
    void CreateDebugUtilsMessenger();
#endif
    void CreateSurface();
    void PickPhysicalDevice();
    void SetQueueFamily(vk::QueueFlags requestedQueueTypes);
    void CreateDevice();
    void CreateVmaAllocator();
    void CreateSwapchain();
    void CreateCommands();
    void CreateSyncStructures();
    void CreatePipelines();

    // compute
    void CreateBackgroundComputeDescriptors();
    void CreateBackgroundComputePipeline();


    void DrawBackground(vk::CommandBuffer cmd);

private:
    // helper functions
    void SetInstanceLayers(
        ::std::vector<::std::string> const& requestedLayers = {});
    void SetInstanceExtensions(
        ::std::vector<::std::string> const& requestedExtensions = {});

    std::vector<std::string> GetSDLRequestedInstanceExtensions() const;

private:
    bool        mStopRendering {false};
    uint32_t    mFrameNum {0};
    SDL_Window* mWindow {nullptr};
    int         mWindowWidth {1600};
    int         mWindowHeight {900};

    ::std::vector<::std::string> mEnabledInstanceLayers {};
    ::std::vector<::std::string> mEnabledInstanceExtensions {};

    vk::Instance       mInstance {};
    VkSurfaceKHR       mSurface {};
    vk::Device         mDevice {};
    vk::PhysicalDevice mPhysicalDevice {};
#ifdef DEBUG
    vk::DebugUtilsMessengerEXT mDebugUtilsMessenger {};
#endif

    std::optional<uint32_t> mGraphicsFamilyIndex;
    uint32_t                mGraphicsQueueCount = 0;
    std::optional<uint32_t> mComputeFamilyIndex;
    uint32_t                mComputeQueueCount = 0;
    std::optional<uint32_t> mTransferFamilyIndex;
    uint32_t                mTransferQueueCount = 0;

    ::std::vector<vk::Queue> mGraphicQueues {};
    ::std::vector<vk::Queue> mComputeQueues {};
    ::std::vector<vk::Queue> mTransferQueues {};

    VmaAllocator mVmaAllocator {};

    vk::SwapchainKHR             mSwapchain {};
    vk::Format                   mSwapchainImageFormat {};
    ::std::vector<vk::Image>     mSwapchainImages {};
    ::std::vector<vk::ImageView> mSwapchainImageViews {};
    vk::Extent2D                 mSwapchainExtent {};

    AllocatedVulkanImage mDrawImage {};

    ::std::vector<vk::DescriptorSet> mDrawImageDescriptors {};
    vk::DescriptorSetLayout          mDrawImageDescriptorLayout {};

    Utils::DeletionQueue mMainDeletionQueue {};

    DescriptorAllocator mMainDescriptorAllocator {};

    // background compute
    vk::Pipeline mBackgroundComputePipeline {};
    vk::PipelineLayout mBackgroundComputePipelineLayout {};
};