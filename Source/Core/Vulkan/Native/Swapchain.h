#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "SyncStructures.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class RenderFrame;
class RenderResource;

class Swapchain {
public:
    Swapchain(VulkanContext* ctx, vk::Format format, vk::Extent2D extent2D);
    ~Swapchain();
    CLASS_MOVABLE_ONLY(Swapchain);

public:
    static constexpr uint64_t WAIT_NEXT_IMAGE_TIME_OUT = 1000000000;

    uint32_t AcquireNextImageIndex(RenderFrame& frame);

    void Present(RenderFrame& frame, vk::Queue queue);

    vk::SwapchainKHR RecreateSwapchain(vk::Extent2D extent,
                                       vk::SwapchainKHR old = VK_NULL_HANDLE);

public:
    void Resize(vk::Extent2D extent);

    vk::SwapchainKHR GetHandle() const;
    vk::Image GetImageHandle(uint32_t index) const;
    vk::ImageView GetImageViewHandle(uint32_t index) const;
    vk::RenderingAttachmentInfo GetColorAttachmentInfo(uint32_t index) const;
    vk::Format GetFormat() const;
    vk::Extent2D GetExtent2D() const;
    uint32_t GetImageCount() const;
    uint32_t GetCurrentImageIndex() const;
    uint32_t GetPrevImageIndex() const;
    RenderResource const& GetCurrentImage() const;

private:
    void SetSwapchainImages();

private:
    VulkanContext* pContex;

    vk::Format mFormat;
    vk::Extent2D mExtent2D {};

    vk::SwapchainCreateInfoKHR mCreateInfo {};
    vk::SwapchainKHR mSwapchain;

    Type_STLVector<RenderResource> mImages {};
    uint32_t mCurrentImageIndex {0};
};

}  // namespace IntelliDesign_NS::Vulkan::Core