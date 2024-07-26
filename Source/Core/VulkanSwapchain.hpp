#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "VulkanSyncStructures.hpp"

class VulkanContext;
class VulkanSemaphore;
class VulkanFence;
class VulkanAllocatedImage;

class VulkanSwapchain {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanSwapchain(Type_SPInstance<VulkanContext> const& ctx,
                    vk::Format format, vk::Extent2D extent2D);
    ~VulkanSwapchain();
    MOVABLE_ONLY(VulkanSwapchain);

public:
    static constexpr uint64_t WAIT_NEXT_IMAGE_TIME_OUT = 1000000000;

    vk::Image AquireNextImageHandle();

    void Present(vk::Queue queue);

    vk::SwapchainKHR RecreateSwapchain(vk::SwapchainKHR old = VK_NULL_HANDLE);

public:
    vk::SwapchainKHR const& GetHandle() const { return mSwapchain; }

    vk::Image const& GetImageHandle(uint32_t index) const;

    vk::ImageView const& GetImageViewHandle(uint32_t index) const;

    vk::Format const& GetFormat() const { return mFormat; }

    vk::Extent2D const& GetExtent2D() const { return mExtent2D; }

    uint32_t GetImageCount() const { return mImages.size(); }

    uint32_t GetCurrentImageIndex() const { return mCurrentImageIndex; }

    vk::Fence const& GetAquireFenceHandle() const {
        return mAquireFence.GetHandle();
    }

    vk::Semaphore const& GetReady4PresentSemHandle() const {
        return mReady4Present.GetHandle();
    }

    vk::Semaphore const& GetReady4RenderSemHandle() const {
        return mReady4Render.GetHandle();
    }

private:
    // TODO: resize window
    void SetSwapchainImages();

private:
    Type_SPInstance<VulkanContext> pContex;

    vk::Format mFormat;
    vk::Extent2D mExtent2D;

    vk::SwapchainCreateInfoKHR mCreateInfo {};
    vk::SwapchainKHR mSwapchain;

    VulkanSemaphore mReady4Present {pContex};
    VulkanSemaphore mReady4Render {pContex};
    VulkanFence mAquireFence {pContex};

    ::std::vector<VulkanAllocatedImage> mImages {};
    uint32_t mCurrentImageIndex {0};
};