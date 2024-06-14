#include "VulkanImage.hpp"

void AllocatedVulkanImage::CreateImage(VmaAllocator            allocator,
                                       VmaAllocationCreateInfo allocCreateInfo,
                                       vk::Extent3D extent, vk::Format format,
                                       vk::ImageUsageFlags usage,
                                       vk::ImageType type, uint32_t mipLevels,
                                       uint32_t arrayLayers) {
    mExtent3D = extent;
    mFormat   = format;

    vk::ImageCreateInfo imageCreateInfo {};
    imageCreateInfo.setImageType(type)
        .setFormat(mFormat)
        .setExtent(mExtent3D)
        .setUsage(usage)
        .setMipLevels(mipLevels)
        .setArrayLayers(arrayLayers);

    vmaCreateImage(allocator, (VkImageCreateInfo*)&imageCreateInfo,
                   &allocCreateInfo, (VkImage*)&mImage, &mAllocation, nullptr);
}

void AllocatedVulkanImage::CreateImageView(vk::Device           device,
                                           vk::ImageAspectFlags aspect,
                                           vk::ImageViewType    type) {
    vk::ImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.setViewType(type)
        .setImage(mImage)
        .setFormat(mFormat)
        .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(aspect));

    mImageView = device.createImageView(imageViewCreateInfo);
}

void AllocatedVulkanImage::DestroyImage(VmaAllocator allocator) {
    vmaDestroyImage(allocator, mImage, mAllocation);
}

void AllocatedVulkanImage::DestroyImageView(vk::Device device) {
    device.destroy(mImageView);
}

void AllocatedVulkanImage::Destroy(vk::Device device, VmaAllocator allocator) {
    DestroyImage(allocator);
    DestroyImageView(device);
}

void AllocatedVulkanImage::TransitionLayout(vk::CommandBuffer cmd,
                                            vk::ImageLayout   newLayout) {
    vk::ImageMemoryBarrier2 imgBarrier {};
    imgBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite |
                          vk::AccessFlagBits2::eMemoryRead)
        .setOldLayout(mLayout)
        .setNewLayout(newLayout)
        .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(
            newLayout == vk::ImageLayout::eDepthAttachmentOptimal
                ? vk::ImageAspectFlagBits::eDepth
                : vk::ImageAspectFlagBits::eColor))
        .setImage(mImage);

    vk::DependencyInfo depInfo {};
    depInfo.setImageMemoryBarrierCount(1u).setImageMemoryBarriers(imgBarrier);

    cmd.pipelineBarrier2(depInfo);
}

void AllocatedVulkanImage::CopyToImage(vk::CommandBuffer cmd,
                                       vk::Image         dstImage,
                                       vk::Extent2D      srcExtent,
                                       vk::Extent2D      dstExtent) {
    vk::ImageBlit2 blitRegion {};
    blitRegion
        .setSrcOffsets(
            {vk::Offset3D {},
             vk::Offset3D {static_cast<int32_t>(srcExtent.width),
                           static_cast<int32_t>(srcExtent.height), 1}})
        .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setDstOffsets(
            {vk::Offset3D {},
             vk::Offset3D {static_cast<int32_t>(dstExtent.width),
                           static_cast<int32_t>(dstExtent.height), 1}})
        .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});

    vk::BlitImageInfo2 blitInfo {};
    blitInfo.setDstImage(dstImage)
        .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcImage(mImage)
        .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setFilter(vk::Filter::eLinear)
        .setRegions(blitRegion);

    cmd.blitImage2(blitInfo);
}

void AllocatedVulkanImage::CopyToImage(vk::CommandBuffer    cmd,
                                       AllocatedVulkanImage dstImage,
                                       vk::Extent2D         srcExtent,
                                       vk::Extent2D         dstExtent) {
    CopyToImage(cmd, dstImage.mImage, srcExtent, dstExtent);
}