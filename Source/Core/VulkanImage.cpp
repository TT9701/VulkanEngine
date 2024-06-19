#include "VulkanImage.hpp"

#include "Engine.hpp"
#include "VulkanBuffer.hpp"

void AllocatedVulkanImage::CreateImage(
    vk::Device device, VmaAllocator allocator,
    VmaAllocationCreateInfo allocCreateInfo, vk::Extent3D extent,
    vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
    bool mipmaped, uint32_t arrayLayers, vk::ImageType type,
    vk::ImageViewType viewType) {
    mExtent3D = extent;
    mFormat   = format;

    uint32_t mipLevels =
        mipmaped ? static_cast<uint32_t>(1 + ::std::floor(::std::log2(std::max(
                                                 extent.width, extent.height))))
                 : 1;

    vk::ImageCreateInfo imageCreateInfo {};
    imageCreateInfo.setImageType(type)
        .setFormat(mFormat)
        .setExtent(mExtent3D)
        .setUsage(usage)
        .setMipLevels(mipLevels)
        .setArrayLayers(arrayLayers);

    vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
                   &allocCreateInfo, reinterpret_cast<VkImage*>(&mImage), &mAllocation, nullptr);

    vk::ImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.setViewType(viewType)
        .setImage(mImage)
        .setFormat(mFormat)
        .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(aspect));

    mImageView = device.createImageView(imageViewCreateInfo);
}

void AllocatedVulkanImage::CreateImage(
    void* data, VulkanEngine* engine, VmaAllocationCreateInfo allocCreateInfo,
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, bool mipmaped, uint32_t arrayLayers,
    vk::ImageType type, vk::ImageViewType viewType) {
    size_t dataSize = extent.width * extent.height * extent.depth * 4;

    AllocatedVulkanBuffer uploadBuffer {};
    uploadBuffer.CreateBuffer(
        engine->GetVmaAllocator(), dataSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    memcpy(uploadBuffer.mInfo.pMappedData, data, dataSize);

    CreateImage(engine->GetVkDevice(), engine->GetVmaAllocator(),
                allocCreateInfo, extent, format, usage, aspect, mipmaped,
                arrayLayers, type, viewType);

    engine->ImmediateSubmit([&](vk::CommandBuffer cmd) {
        TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
        vk::BufferImageCopy copyRegion {};
        copyRegion
            .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(extent);

        cmd.copyBufferToImage(uploadBuffer.mBuffer, mImage,
                              vk::ImageLayout::eTransferDstOptimal, copyRegion);

        TransitionLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    uploadBuffer.Destroy();
}

void AllocatedVulkanImage::Destroy(vk::Device device, VmaAllocator allocator) {
    vmaDestroyImage(allocator, mImage, mAllocation);
    device.destroy(mImageView);
}

void AllocatedVulkanImage::TransitionLayout(vk::CommandBuffer cmd,
                                            vk::ImageLayout   newLayout) {
    Utils::TransitionImageLayout(cmd, mImage, mLayout, newLayout);
    mLayout = newLayout;
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