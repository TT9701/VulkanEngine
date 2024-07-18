#include "VulkanImage.hpp"

#include "Engine.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanDevice.hpp"
#include "VulkanMemoryAllocator.hpp"

VulkanAllocatedImage::VulkanAllocatedImage(
    Type_SPInstance<VulkanDevice> const& device,
    Type_SPInstance<VulkanMemoryAllocator> const& allocator,
    VmaAllocationCreateFlags flags, vk::Extent3D extent, vk::Format format,
    vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect, void* data,
    VulkanEngine* engine, uint32_t mipmapLevel, uint32_t arrayLayers,
    vk::ImageType type, vk::ImageViewType viewType)
    : pDevice(device),
      pAllocator(allocator),
      mExtent3D(extent),
      mFormat(format),
      pEngine(engine),
      mMipmapLevel(mipmapLevel),
      mArrayLayerCount(arrayLayers),
      mImage(CreateImage(data, flags, usage, type)),
      mImageView(CreateImageView(aspect, viewType)) {}

VulkanAllocatedImage::~VulkanAllocatedImage() {
    vmaDestroyImage(pAllocator->GetHandle(), mImage, mAllocation);
    pDevice->GetHandle().destroy(mImageView);
}

// void VulkanAllocatedImage::CreateImage(
//     VmaAllocationCreateFlags flags,
//     vk::ImageUsageFlags usage,
//     vk::ImageAspectFlags aspect, int mipmapLevel, uint32_t arrayLayers,
//     vk::ImageType type, vk::ImageViewType viewType) {
//     mExtent3D = extent;
//     mFormat = format;
//
//     uint32_t mipLevels =
//         mipmaped ? static_cast<uint32_t>(1
//                                          + ::std::floor(::std::log2(std::max(
//                                              extent.width, extent.height))))
//                  : 1;
//
//     vk::ImageCreateInfo imageCreateInfo {};
//     imageCreateInfo.setImageType(type)
//         .setFormat(mFormat)
//         .setExtent(mExtent3D)
//         .setUsage(usage)
//         .setMipLevels(mipLevels)
//         .setArrayLayers(arrayLayers);
//
//     /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
//     VmaAllocationCreateInfo allocCreateInfo {};
//     allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
//     allocCreateInfo.flags = flags;
//
//     vmaCreateImage(allocator,
//                    reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
//                    &allocCreateInfo, reinterpret_cast<VkImage*>(&mImage),
//                    &mAllocation, nullptr);
//
//     vk::ImageViewCreateInfo imageViewCreateInfo {};
//     imageViewCreateInfo.setViewType(viewType)
//         .setImage(mImage)
//         .setFormat(mFormat)
//         .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(aspect));
//
//     mImageView = device.createImageView(imageViewCreateInfo);
// }

void VulkanAllocatedImage::TransitionLayout(vk::CommandBuffer cmd,
                                            vk::ImageLayout newLayout) {
    Utils::TransitionImageLayout(cmd, mImage, mLayout, newLayout);
    mLayout = newLayout;
}

void VulkanAllocatedImage::CopyToImage(vk::CommandBuffer cmd,
                                       vk::Image dstImage,
                                       vk::Extent2D srcExtent,
                                       vk::Extent2D dstExtent) {
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

void VulkanAllocatedImage::CopyToImage(vk::CommandBuffer cmd,
                                       VulkanAllocatedImage dstImage,
                                       vk::Extent2D srcExtent,
                                       vk::Extent2D dstExtent) {
    CopyToImage(cmd, dstImage.mImage, srcExtent, dstExtent);
}

vk::Image VulkanAllocatedImage::CreateImage(void* data,
                                            VmaAllocationCreateFlags flags,
                                            vk::ImageUsageFlags usage,
                                            vk::ImageType type) {
    VkImage image {};

    vk::ImageCreateInfo imageCreateInfo {};
    imageCreateInfo.setImageType(type)
        .setFormat(mFormat)
        .setExtent(mExtent3D)
        .setUsage(usage)
        .setMipLevels(mMipmapLevel)
        .setArrayLayers(mArrayLayerCount);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo allocCreateInfo {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = flags;

    vmaCreateImage(pAllocator->GetHandle(),
                   reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
                   &allocCreateInfo, &image, &mAllocation, nullptr);
    if (data) {
        size_t dataSize = static_cast<size_t>(mExtent3D.width)
                        * mExtent3D.height * mExtent3D.depth * 4;

        VulkanAllocatedBuffer uploadBuffer {
            pAllocator, dataSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT};

        memcpy(uploadBuffer.GetAllocationInfo().pMappedData, data, dataSize);

        pEngine->ImmediateSubmit([&](vk::CommandBuffer cmd) {
            TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
            vk::BufferImageCopy copyRegion {};
            copyRegion
                .setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(mExtent3D);

            cmd.copyBufferToImage(uploadBuffer.GetHandle(), image,
                                  vk::ImageLayout::eTransferDstOptimal,
                                  copyRegion);

            TransitionLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        });
    }

    return image;
}

vk::ImageView VulkanAllocatedImage::CreateImageView(
    vk::ImageAspectFlags aspect, vk::ImageViewType viewType) {

    vk::ImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.setViewType(viewType)
        .setImage(mImage)
        .setFormat(mFormat)
        .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(aspect));

    return pDevice->GetHandle().createImageView(imageViewCreateInfo);
}