#pragma once

#include <Core/Utilities/VulkanUtilities.hpp>

class AllocatedVulkanImage {
public:
    void CreateImage(VmaAllocator            allocator,
                     VmaAllocationCreateInfo allocCreateInfo,
                     vk::Extent3D extent, vk::Format format,
                     vk::ImageUsageFlags usage,
                     vk::ImageType       type = vk::ImageType::e2D,
                     bool mipmaped = false, uint32_t arrayLayers = 1);

    // must executed afer CreateImage()
    void CreateImageView(vk::Device device, vk::ImageAspectFlags aspect,
                         vk::ImageViewType type = vk::ImageViewType::e2D);

    void DestroyImage(VmaAllocator allocator);
    void DestroyImageView(vk::Device device);
    void Destroy(vk::Device device, VmaAllocator allocator);

    void TransitionLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    void CopyToImage(vk::CommandBuffer cmd, vk::Image dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);
    void CopyToImage(vk::CommandBuffer cmd, AllocatedVulkanImage dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);

    vk::Image       mImage {};
    vk::ImageView   mImageView {};
    VmaAllocation   mAllocation {};
    vk::Extent3D    mExtent3D {};
    vk::Format      mFormat {};
    vk::ImageLayout mLayout {vk::ImageLayout::eUndefined};
};