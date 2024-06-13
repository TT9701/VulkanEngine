#pragma once

#include <vulkan/vulkan.hpp>

#include "Utilities/VulkanUtilities.hpp"

struct SDL_Window;

class VulkanEngine {
public:
    void Init();
    void Run();
    void Cleanup();

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

private:
    // helper functions
    void SetInstanceLayers(::std::vector<::std::string> const& requestedLayers = {});
    void SetInstanceExtensions(::std::vector<::std::string> const& requestedExtensions = {});

    std::vector<std::string> GetSDLRequestedInstanceExtensions() const;

private:
    bool        mStopRendering {false};
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
};