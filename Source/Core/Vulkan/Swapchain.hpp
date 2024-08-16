#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "SyncStructures.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Semaphore;
class Fence;
class RenderResource;

class Swapchain {
public:
    Swapchain(Context* ctx, vk::Format format,
                    vk::Extent2D extent2D);
    ~Swapchain();
    MOVABLE_ONLY(Swapchain);

public:
    static constexpr uint64_t WAIT_NEXT_IMAGE_TIME_OUT = 1000000000;

    uint32_t AcquireNextImageIndex();

    void Present(vk::Queue queue);

    vk::SwapchainKHR RecreateSwapchain(vk::SwapchainKHR old = VK_NULL_HANDLE);

public:
    vk::SwapchainKHR GetHandle() const { return mSwapchain; }

    vk::Image GetImageHandle(uint32_t index) const;

    vk::ImageView GetImageViewHandle(uint32_t index) const;

    vk::Format GetFormat() const { return mFormat; }

    vk::Extent2D GetExtent2D() const { return mExtent2D; }

    uint32_t GetImageCount() const { return mImages.size(); }

    uint32_t GetCurrentImageIndex() const { return mCurrentImageIndex; }

    RenderResource const& GetCurrentImage() const {
        return mImages[mCurrentImageIndex];
    }

    vk::Fence GetAquireFenceHandle() const { return mAcquireFence.GetHandle(); }

    vk::Semaphore GetReady4PresentSemHandle() const {
        return mReady4Present.GetHandle();
    }

    vk::Semaphore GetReady4RenderSemHandle() const {
        return mReady4Render.GetHandle();
    }

private:
    // TODO: resize window
    void SetSwapchainImages();

private:
    Context* pContex;

    vk::Format mFormat;
    vk::Extent2D mExtent2D;

    vk::SwapchainCreateInfoKHR mCreateInfo {};
    vk::SwapchainKHR mSwapchain;

    Semaphore mReady4Present {pContex};
    Semaphore mReady4Render {pContex};
    Fence mAcquireFence {pContex};

    ::std::vector<RenderResource> mImages {};
    uint32_t mCurrentImageIndex {0};
};

}  // namespace IntelliDesign_NS::Vulkan::Core