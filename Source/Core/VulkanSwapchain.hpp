#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanHelper.hpp"

class VulkanSurface;
class VulkanDevice;

class VulkanSwapchain {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanSwapchain(Type_SPInstance<VulkanDevice> const& device,
                    Type_SPInstance<VulkanSurface> const& surface,
                    vk::Format format, vk::Extent2D extent2D);
    ~VulkanSwapchain();

    vk::SwapchainKHR RecreateSwapchain(vk::SwapchainKHR old);

public:
    vk::SwapchainKHR const& GetHandle() const { return mSwapchain; }

    ::std::vector<vk::Image> const& GetImages() const { return mImages; }

    ::std::vector<vk::ImageView> const& GetImageViews() const {
        return mImageViews;
    }

    vk::Format const& GetFormat() const { return mFormat; }

    vk::Extent2D const& GetExtent2D() const { return mExtent2D; }

private:
    // TODO: resize window
    void SetSwapchainImages();
    void CreateSwapchainImageViews();

private:
    Type_SPInstance<VulkanDevice> pDevice;
    Type_SPInstance<VulkanSurface> pSurface;

    vk::Format mFormat;
    vk::Extent2D mExtent2D;

    vk::SwapchainCreateInfoKHR mCreateInfo {};
    vk::SwapchainKHR mSwapchain;

    ::std::vector<vk::Image> mImages {};
    ::std::vector<vk::ImageView> mImageViews {};
};