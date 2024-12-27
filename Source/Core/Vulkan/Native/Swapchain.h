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
    Swapchain(VulkanContext& ctx, vk::Format format, vk::Extent2D extent2D);
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
    RenderResource const& GetImage(uint32_t index) const;
    Type_STLVector<RenderResource> const& GetImages() const;

private:
    void SetSwapchainImages();

private:
    VulkanContext& mContex;

    vk::Format mFormat;
    vk::Extent2D mExtent2D {};

    vk::SwapchainCreateInfoKHR mCreateInfo {};
    vk::SwapchainKHR mSwapchain;

    Type_STLVector<RenderResource> mImages {};
    uint32_t mCurrentImageIndex {0};
};

struct HPPSwapchainProperties {
    vk::SwapchainKHR oldSwapchain;
    uint32_t imageCount {3};
    vk::Extent2D extent;
    vk::SurfaceFormatKHR surfaceFormat;
    uint32_t arrayLayers;
    vk::ImageUsageFlags imageUsage;
    vk::SurfaceTransformFlagBitsKHR preTransform;
    vk::CompositeAlphaFlagBitsKHR compositeAlpha;
    vk::PresentModeKHR presentMode;
};

class HPPSwapchain {
public:
    HPPSwapchain(HPPSwapchain& old, vk::Extent2D const& extent);

    HPPSwapchain(
        VulkanContext& context, vk::PresentModeKHR presentMode,
        std::vector<vk::PresentModeKHR> const& presentModePriorityList =
            {vk::PresentModeKHR::eFifo, vk::PresentModeKHR::eMailbox},
        std::vector<vk::SurfaceFormatKHR> const& surfaceFormatPriorityList =
            {{vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear},
             {vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear}},
        vk::Extent2D const& extent = {}, uint32_t imageCount = 3,
        vk::SurfaceTransformFlagBitsKHR transform =
            vk::SurfaceTransformFlagBitsKHR::eIdentity,
        std::set<vk::ImageUsageFlagBits> const& imageUsageFlags =
            {vk::ImageUsageFlagBits::eColorAttachment,
             vk::ImageUsageFlagBits::eTransferSrc},
        vk::SwapchainKHR old = VK_NULL_HANDLE);

    CLASS_NO_COPY(HPPSwapchain);

    HPPSwapchain(HPPSwapchain&& other) noexcept;

    HPPSwapchain& operator=(HPPSwapchain&&) = delete;

    ~HPPSwapchain();

    static constexpr uint64_t WAIT_NEXT_IMAGE_TIME_OUT = 1000000000;

    bool IsValid() const;

    vk::SwapchainKHR GetHandle() const;

    std::pair<vk::Result, uint32_t> AcquireNextImage(
        vk::Semaphore imageAcquiredSemaphore, vk::Fence fence = nullptr) const;

    vk::Extent2D const& GetExtent() const;

    vk::Format GetFormat() const;

    std::vector<vk::Image> const& GetImages() const;

    vk::ImageUsageFlags GetUsage() const;

    vk::PresentModeKHR GetPresentMode() const;

private:
    VulkanContext& mContext;

    vk::SwapchainKHR mHandle;

    std::vector<vk::Image> mImages;

    HPPSwapchainProperties mProperties;

    std::vector<vk::PresentModeKHR> mPresentModePriorityList;

    std::vector<vk::SurfaceFormatKHR> mSurfaceFormatPriorityList;

    std::set<vk::ImageUsageFlagBits> mImageUsageFlag;
};

}  // namespace IntelliDesign_NS::Vulkan::Core