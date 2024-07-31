#pragma once

#include <Core/Utilities/VulkanUtilities.hpp>
#include "Core/Utilities/Defines.hpp"

class VulkanContext;
class VulkanMemoryAllocator;
class VulkanEngine;

class VulkanImage {
public:
    VulkanImage(VulkanContext* ctx, VmaAllocationCreateFlags flags,
                vk::Extent3D extent, vk::Format format,
                vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
                void* data = nullptr, VulkanEngine* engine = nullptr,
                uint32_t mipmapLevel = 1, uint32_t arrayLayers = 1,
                vk::ImageType     type     = vk::ImageType::e2D,
                vk::ImageViewType viewType = vk::ImageViewType::e2D);

    VulkanImage(VulkanContext* ctx, vk::Image image, vk::Extent3D extent,
                vk::Format format, vk::ImageAspectFlags aspect,
                uint32_t          arrayLayers = 1,
                vk::ImageViewType viewType    = vk::ImageViewType::e2D);

    ~VulkanImage();

    MOVABLE_ONLY(VulkanImage);

public:
    void TransitionLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    void CopyToImage(vk::CommandBuffer cmd, vk::Image dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);

    void CopyToImage(vk::CommandBuffer cmd, VulkanImage dstImage,
                     vk::Extent2D srcExtent, vk::Extent2D dstExtent);

    void UploadData(void* data);

public:
    vk::Extent3D GetExtent3D() const { return mExtent3D; }

    vk::Format GetFormat() const { return mFormat; }

    uint32_t GetMipmapLevel() const { return mMipmapLevel; }

    uint32_t GetArrayLayerCount() const { return mArrayLayerCount; }

    vk::ImageLayout GetLayout() const { return mLayout; }

    vk::Image GetHandle() const { return mImage; }

    vk::ImageView GetViewHandle() const { return mImageView; }

private:
    vk::Image CreateImage(void* data, VmaAllocationCreateFlags flags,
                          vk::ImageUsageFlags usage, vk::ImageType type);

    vk::ImageView CreateImageView(vk::ImageAspectFlags aspect,
                                  vk::ImageViewType    viewType);

private:
    VulkanContext* pContex;

    vk::Extent3D mExtent3D;
    vk::Format   mFormat;

    bool mOwnsImage;

    VulkanEngine* pEngine;

    uint32_t        mMipmapLevel;
    uint32_t        mArrayLayerCount;
    vk::ImageLayout mLayout {vk::ImageLayout::eUndefined};

    VmaAllocation mAllocation {};

    vk::Image     mImage {};
    vk::ImageView mImageView {};
};