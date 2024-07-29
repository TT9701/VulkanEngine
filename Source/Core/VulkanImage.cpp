#include "VulkanImage.hpp"

#include "Engine.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanContext.hpp"
#include "VulkanMemoryAllocator.hpp"

VulkanAllocatedImage::VulkanAllocatedImage(
    SharedPtr<VulkanContext> const& ctx, VmaAllocationCreateFlags flags,
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, void* data, VulkanEngine* engine,
    uint32_t mipmapLevel, uint32_t arrayLayers, vk::ImageType type,
    vk::ImageViewType viewType)
    : pContex(ctx),
      mExtent3D(extent),
      mFormat(format),
      mOwnsImage(true),
      pEngine(engine),
      mMipmapLevel(mipmapLevel),
      mArrayLayerCount(arrayLayers),
      mImage(CreateImage(data, flags, usage, type)),
      mImageView(CreateImageView(aspect, viewType)) {
    UploadData(data);
}

VulkanAllocatedImage::VulkanAllocatedImage(
    SharedPtr<VulkanContext> const& ctx, vk::Image image,
    vk::Extent3D extent, vk::Format format, vk::ImageAspectFlags aspect,
    uint32_t arrayLayers, vk::ImageViewType viewType)
    : pContex(ctx),
      mExtent3D(extent),
      mFormat(format),
      mOwnsImage(false),
      pEngine(nullptr),
      mMipmapLevel(1u),
      mArrayLayerCount(arrayLayers),
      mImage(image),
      mImageView(CreateImageView(aspect, viewType)) {}

VulkanAllocatedImage::~VulkanAllocatedImage() {
    if (mOwnsImage)
        vmaDestroyImage(pContex->GetVmaAllocator()->GetHandle(), mImage, mAllocation);
    pContex->GetDevice()->GetHandle().destroy(mImageView);
}

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

void VulkanAllocatedImage::UploadData(void* data) {
    if (data) {
        size_t dataSize = static_cast<size_t>(mExtent3D.width)
                        * mExtent3D.height * mExtent3D.depth * 4;

        VulkanAllocatedBuffer uploadBuffer {
            pContex->GetVmaAllocator(), dataSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT};

        memcpy(uploadBuffer.GetAllocationInfo().pMappedData, data, dataSize);

        pEngine->GetImmediateSubmitManager()->Submit(
            [&](vk::CommandBuffer cmd) {
                TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
                vk::BufferImageCopy copyRegion {};
                copyRegion
                    .setImageSubresource(
                        {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                    .setImageExtent(mExtent3D);

                cmd.copyBufferToImage(uploadBuffer.GetHandle(), mImage,
                                      vk::ImageLayout::eTransferDstOptimal,
                                      copyRegion);

                TransitionLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
            });
    }
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

    vmaCreateImage(pContex->GetVmaAllocator()->GetHandle(),
                   reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
                   &allocCreateInfo, &image, &mAllocation, nullptr);

    return image;
}

vk::ImageView VulkanAllocatedImage::CreateImageView(
    vk::ImageAspectFlags aspect, vk::ImageViewType viewType) {

    vk::ImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.setViewType(viewType)
        .setImage(mImage)
        .setFormat(mFormat)
        .setSubresourceRange(Utils::GetDefaultImageSubresourceRange(aspect));

    return pContex->GetDevice()->GetHandle().createImageView(
        imageViewCreateInfo);
}