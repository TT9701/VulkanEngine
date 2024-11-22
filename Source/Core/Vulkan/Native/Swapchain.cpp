#include "Swapchain.h"

#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/Context.h"
#include "Core/Vulkan/Manager/RenderFrame.h"
#include "RenderResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

Swapchain::Swapchain(Context* ctx, vk::Format format, vk::Extent2D extent2D)
    : pContex(ctx), mFormat(format), mSwapchain(RecreateSwapchain(extent2D)) {
    SetSwapchainImages();

    DBG_LOG_INFO(
        "Vulkan Swapchain Created. PresentMode: %s. \n\t\t\t    "
        "Swapchain Image Count: %d",
        vk::to_string(mCreateInfo.presentMode).c_str(), mImages.size());
}

Swapchain::~Swapchain() {
    pContex->GetDeviceHandle().destroy(mSwapchain);
}

uint32_t Swapchain::AcquireNextImageIndex(RenderFrame& frame) {
    frame.Reset();

    VK_CHECK(pContex->GetDeviceHandle().acquireNextImageKHR(
        mSwapchain, WAIT_NEXT_IMAGE_TIME_OUT,
        frame.GetReady4RenderSemaphore().GetHandle(), VK_NULL_HANDLE,
        &mCurrentImageIndex));

    return mCurrentImageIndex;
}

void Swapchain::Present(RenderFrame& frame, vk::Queue queue) {
    auto sem = frame.GetReady4PresentSemaphore().GetHandle();

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(sem)
        .setImageIndices(mCurrentImageIndex);

    VK_CHECK(queue.presentKHR(presentInfo));
}

vk::SwapchainKHR Swapchain::RecreateSwapchain(vk::Extent2D extent,
                                              vk::SwapchainKHR old) {
    mExtent2D = extent;
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

    auto handle = pContex->GetDeviceHandle().createSwapchainKHR(mCreateInfo);
    pContex->SetName(handle, "Default Swapchain");
    return handle;
}

vk::SwapchainKHR Swapchain::GetHandle() const {
    return mSwapchain;
}

vk::Image Swapchain::GetImageHandle(uint32_t index) const {
    return mImages[index].GetTexHandle();
}

vk::ImageView Swapchain::GetImageViewHandle(uint32_t index) const {
    return mImages[index].GetTexViewHandle("Color-Whole");
}

vk::RenderingAttachmentInfo Swapchain::GetColorAttachmentInfo(
    uint32_t index) const {
    vk::RenderingAttachmentInfo info {};
    info.setImageView(mImages[index].GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    return info;
}

vk::Format Swapchain::GetFormat() const {
    return mFormat;
}

vk::Extent2D Swapchain::GetExtent2D() const {
    return mExtent2D;
}

uint32_t Swapchain::GetImageCount() const {
    return mImages.size();
}

uint32_t Swapchain::GetCurrentImageIndex() const {
    return mCurrentImageIndex;
}

uint32_t Swapchain::GetPrevImageIndex() const {
    return (mCurrentImageIndex - 1) % 3;
}

RenderResource const& Swapchain::GetCurrentImage() const {
    return mImages[mCurrentImageIndex];
}

void Swapchain::Resize(vk::Extent2D extent) {
    pContex->GetDeviceHandle().waitIdle();
    auto newSP = RecreateSwapchain(extent, mSwapchain);
    pContex->GetDeviceHandle().destroy(mSwapchain);
    mSwapchain = newSP;
    SetSwapchainImages();
}

void Swapchain::SetSwapchainImages() {
    mImages.clear();
    auto images = pContex->GetDeviceHandle().getSwapchainImagesKHR(mSwapchain);
    mImages.reserve(images.size());
    for (auto& img : images) {
        mImages.emplace_back(
            pContex->GetDevice(), img, RenderResource::Type::Texture2D, mFormat,
            vk::Extent3D {mExtent2D.width, mExtent2D.height, 1}, 1, 1);

        pContex->SetName(mImages.back().GetTexHandle(), "Swapchain Images");

        mImages.back().CreateTexView("Color-Whole",
                                     vk::ImageAspectFlagBits::eColor);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core