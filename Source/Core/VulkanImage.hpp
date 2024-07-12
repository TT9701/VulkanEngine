#pragma once

#include <Core/Utilities/VulkanUtilities.hpp>

#include "VulkanHelper.hpp"

class VulkanDevice;
class VulkanMemoryAllocator;
class VulkanEngine;

class VulkanAllocatedImage {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);
public:
    // VulkanAllocatedImage();
    ~VulkanAllocatedImage();

public:
    void CreateImage(vk::Device device, VmaAllocator allocator,
                     VmaAllocationCreateFlags flags, vk::Extent3D extent,
                     vk::Format format, vk::ImageUsageFlags usage,
                     vk::ImageAspectFlags aspect, bool mipmaped = false,
                     uint32_t arrayLayers = 1,
                     vk::ImageType type = vk::ImageType::e2D,
                     vk::ImageViewType viewType = vk::ImageViewType::e2D);

    void CreateImage(void* data, VulkanEngine* engine,
                     VmaAllocationCreateFlags flags, vk::Extent3D extent,
                     vk::Format format, vk::ImageUsageFlags usage,
                     vk::ImageAspectFlags aspect, bool mipmaped = false,
                     uint32_t arrayLayers = 1,
                     vk::ImageType type = vk::ImageType::e2D,
                     vk::ImageViewType viewType = vk::ImageViewType::e2D);

    void Destroy(vk::Device device, VmaAllocator allocator);

    void TransitionLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    void CopyToImage(vk::CommandBuffer cmd, vk::Image dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);
    void CopyToImage(vk::CommandBuffer cmd, VulkanAllocatedImage dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);

    vk::Image mImage {};
    vk::ImageView mImageView {};
    VmaAllocation mAllocation {};
    vk::Extent3D mExtent3D {};
    vk::Format mFormat {};
    vk::ImageLayout mLayout {vk::ImageLayout::eUndefined};
};