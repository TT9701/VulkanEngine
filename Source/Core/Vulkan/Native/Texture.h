#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Sampler;
class DescriptorManager;
class ImageView;

class Texture {
    using Type_ImageViews = Type_STLUnorderedMap_String<SharedPtr<ImageView>>;

public:
    enum class Type {
        Texture1D,
        Texture2D,
        Texture3D,
        TextureCube,
        Texture2DMultisample,
    };

    // No data source
    Texture(VulkanContext& context, Type type, vk::Format format,
            vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
            uint32_t arraySize, uint32_t sampleCount);

    // from swapchain
    Texture(VulkanContext& context, vk::Image handle, Type type,
            vk::Format format, vk::Extent3D extent, uint32_t arraySize,
            uint32_t sampleCount);

    ~Texture();

public:
    void CreateImageView(const char* name, vk::ImageAspectFlags aspect,
                         uint32_t mostDetailedMip = 0,
                         uint32_t mipCount = vk::RemainingMipLevels,
                         uint32_t firstArray = 0,
                         uint32_t arraySize = vk::RemainingArrayLayers);

public:
    uint32_t GetWidth(uint32_t mipLevel = 0) const;
    uint32_t GetHeight(uint32_t mipLevel = 0) const;
    uint32_t GetDepth(uint32_t mipLevel = 0) const;
    uint32_t GetMipCount() const;
    uint32_t GetSampleCount() const;
    uint32_t GetArraySize() const;
    vk::ImageUsageFlags GetUsage() const;
    vk::Format GetFormat() const;
    vk::Image GetHandle() const;
    Type GetType() const;
    ImageView* GetView(const char* name = nullptr) const;
    vk::ImageView GetViewHandle(const char* name = nullptr) const;

    void SetName(const char* name) const;

    void Resize(vk::Extent2D extent);

private:
    vk::Image CreateImage();

    void Destroy();

private:
    VulkanContext& mContext;

    bool bOwnsImage;
    Type mType;

    vk::Format mFormat;
    vk::Extent3D mExtent3D;
    vk::ImageUsageFlags mUsageFlags;

    uint32_t mMipLevels;
    uint32_t mArraySize;
    uint32_t mSampleCount;

    VmaAllocation mAllocation {};
    VmaAllocationInfo mAllocationInfo {};

    vk::Image mHandle;

    Type_ImageViews mViews;
};

class ImageView {
public:
    ImageView(VulkanContext& context, vk::ImageSubresourceRange range,
              vk::Image imageHandle, vk::Format format, vk::ImageViewType type);

    ~ImageView();

    vk::ImageView GetHandle() const;
    Type_STLString const& GetName() const;

    void Destroy();

    friend class Texture;

private:
    vk::ImageView CreateImageView() const;

private:
    VulkanContext& mContext;
    vk::ImageSubresourceRange mRange;
    vk::Image mImageHandle;
    vk::Format mFormat;
    vk::ImageViewType mType;

    vk::ImageView mHandle;

    Type_STLString mName {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core