#include "VulkanSwapchain.hpp"

#include "VulkanDevice.hpp"
#include "VulkanSurface.hpp"

VulkanSwapchain::VulkanSwapchain(Type_SPInstance<VulkanDevice> const& device,
                                 Type_SPInstance<VulkanSurface> const& surface,
                                 vk::Format format, vk::Extent2D extent2D)
    : pDevice(device),
      pSurface(surface),
      mFormat(format),
      mExtent2D(extent2D),
      mSwapchain(RecreateSwapchain(VK_NULL_HANDLE)) {
    SetSwapchainImages();
    CreateSwapchainImageViews();

    DBG_LOG_INFO(
        "Vulkan Swapchain Created. PresentMode: %s. \n\t\t\t    "
        "Swapchain Image Count: %d",
        vk::to_string(mCreateInfo.presentMode).c_str(), mImages.size());
}

VulkanSwapchain::~VulkanSwapchain() {
    for (auto& view : mImageViews)
        pDevice->GetHandle().destroy(view);
    pDevice->GetHandle().destroy(mSwapchain);
}

vk::SwapchainKHR VulkanSwapchain::RecreateSwapchain(vk::SwapchainKHR old) {
    mCreateInfo.setSurface(pSurface->GetHandle())
        .setMinImageCount(3u)
        .setImageFormat(mFormat)
        .setImageExtent(mExtent2D)
        .setImageArrayLayers(1u)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment
                       | vk::ImageUsageFlagBits::eTransferDst)
        .setPresentMode(vk::PresentModeKHR::eMailbox)
        .setClipped(vk::True)
        .setOldSwapchain(VK_NULL_HANDLE);

    return pDevice->GetHandle().createSwapchainKHR(mCreateInfo);
}

void VulkanSwapchain::SetSwapchainImages() {
    mImages = pDevice->GetHandle().getSwapchainImagesKHR(mSwapchain);
}

void VulkanSwapchain::CreateSwapchainImageViews() {
    mImageViews.resize(mImages.size());
    for (int i = 0; i < mImages.size(); ++i) {
        vk::ImageViewCreateInfo imgViewCreateInfo {};
        imgViewCreateInfo.setImage(mImages[i])
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(mFormat)
            .setSubresourceRange(vk::ImageSubresourceRange {
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        mImageViews[i] =
            pDevice->GetHandle().createImageView(imgViewCreateInfo);
    }
}
