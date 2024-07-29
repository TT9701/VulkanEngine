#include "VulkanSwapchain.hpp"

#include "Utilities/Logger.hpp"
#include "VulkanContext.hpp"
#include "VulkanImage.hpp"

VulkanSwapchain::VulkanSwapchain(SharedPtr<VulkanContext> const& ctx,
                                 vk::Format format, vk::Extent2D extent2D)
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
    pContex->GetDevice()->GetHandle().destroy(mSwapchain);
}

vk::Image VulkanSwapchain::AquireNextImageHandle() {
    VK_CHECK(pContex->GetDevice()->GetHandle().waitForFences(
        mAquireFence.GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS));

    pContex->GetDevice()->GetHandle().resetFences(mAquireFence.GetHandle());

    VK_CHECK(pContex->GetDevice()->GetHandle().acquireNextImageKHR(
        mSwapchain, WAIT_NEXT_IMAGE_TIME_OUT, mReady4Render.GetHandle(),
        VK_NULL_HANDLE, &mCurrentImageIndex));

    return mImages[mCurrentImageIndex].GetHandle();
}

void VulkanSwapchain::Present(vk::Queue queue) {
    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(mReady4Present.GetHandle())
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

    return pContex->GetDevice()->GetHandle().createSwapchainKHR(mCreateInfo);
}

vk::Image const& VulkanSwapchain::GetImageHandle(uint32_t index) const {
    return mImages[index].GetHandle();
}

vk::ImageView const& VulkanSwapchain::GetImageViewHandle(uint32_t index) const {
    return mImages[index].GetViewHandle();
}

void VulkanSwapchain::SetSwapchainImages() {
    auto images =
        pContex->GetDevice()->GetHandle().getSwapchainImagesKHR(mSwapchain);
    mImages.reserve(images.size());
    for (auto& img : images) {
        mImages.emplace_back(
            pContex, img, vk::Extent3D {mExtent2D.width, mExtent2D.height, 1},
            mFormat, vk::ImageAspectFlagBits::eColor);
    }
}