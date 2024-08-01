#include "VulkanSwapchain.hpp"

#include "Core/Utilities/Logger.hpp"
#include "VulkanContext.hpp"
#include "VulkanImage.hpp"

VulkanSwapchain::VulkanSwapchain(VulkanContext* ctx, vk::Format format,
                                 vk::Extent2D extent2D)
    : pContex(ctx),
      mFormat(format),
      mExtent2D(extent2D),
      mSwapchain(RecreateSwapchain()) {
    SetSwapchainImages();

    DBG_LOG_INFO(
        "Vulkan Swapchain Created. PresentMode: %s. \n\t\t\t    "
        "Swapchain Image Count: %d",
        vk::to_string(mCreateInfo.presentMode).c_str(), mImages.size());
}

VulkanSwapchain::~VulkanSwapchain() {
    pContex->GetDeviceHandle().destroy(mSwapchain);
}

uint32_t VulkanSwapchain::AcquireNextImageIndex() {
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        mAcquireFence.GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS));

    pContex->GetDeviceHandle().resetFences(mAcquireFence.GetHandle());

    VK_CHECK(pContex->GetDeviceHandle().acquireNextImageKHR(
        mSwapchain, WAIT_NEXT_IMAGE_TIME_OUT, mReady4Render.GetHandle(),
        mAcquireFence.GetHandle(), &mCurrentImageIndex));

    return mCurrentImageIndex;
}

void VulkanSwapchain::Present(vk::Queue queue) {
    auto sem = mReady4Present.GetHandle();

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(sem)
        .setImageIndices(mCurrentImageIndex);

    VK_CHECK(queue.presentKHR(presentInfo));
}

vk::SwapchainKHR VulkanSwapchain::RecreateSwapchain(vk::SwapchainKHR old) {
    mCreateInfo.setSurface(pContex->GetSurface()->GetHandle())
        .setMinImageCount(3u)
        .setImageFormat(mFormat)
        .setImageExtent(mExtent2D)
        .setImageArrayLayers(1u)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment
                       | vk::ImageUsageFlagBits::eTransferDst)
        .setPresentMode(vk::PresentModeKHR::eMailbox)
        .setClipped(vk::True)
        .setOldSwapchain(old);

    return pContex->GetDeviceHandle().createSwapchainKHR(mCreateInfo);
}

vk::Image VulkanSwapchain::GetImageHandle(uint32_t index) const {
    return mImages[index].GetHandle();
}

vk::ImageView VulkanSwapchain::GetImageViewHandle(uint32_t index) const {
    return mImages[index].GetViewHandle();
}

void VulkanSwapchain::SetSwapchainImages() {
    auto images = pContex->GetDeviceHandle().getSwapchainImagesKHR(mSwapchain);
    mImages.reserve(images.size());
    for (auto& img : images) {
        mImages.emplace_back(
            pContex, img, vk::Extent3D {mExtent2D.width, mExtent2D.height, 1},
            mFormat, vk::ImageAspectFlagBits::eColor);
    }
}