#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Device;
class MemoryAllocator;
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
    Texture(Device* device, MemoryAllocator* allocator, Type type,
            vk::Format format, vk::Extent3D extent, vk::ImageUsageFlags usage,
            uint32_t mipLevels, uint32_t arraySize, uint32_t sampleCount);

    // from swapchain
    Texture(Device* device, vk::Image handle, Type type, vk::Format format,
            vk::Extent3D extent, uint32_t arraySize, uint32_t sampleCount);

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
    vk::Format GetFormat() const;
    vk::Image GetHandle() const;
    Type GetType() const;
    ImageView* GetView(const char* name) const;
    vk::ImageView GetViewHandle(const char* name) const;

    void SetName(const char* name) const;

    void AllocateDescriptor(DescriptorManager* manager, uint32_t binding,
                            const char* descSetName, vk::DescriptorType type,
                            const char* viewName,
                            Sampler* sampler = nullptr) const;

    void Resize(vk::Extent2D extent);

private:
    vk::Image CreateImage();

    void Destroy();

private:
    Device* pDevice;
    MemoryAllocator* pAllocator;

    bool bOwnsImage;
    Type mType;

    vk::Format mFormat;
    vk::Extent3D mExtent3D;
    vk::ImageUsageFlags mUsageFlags;

    uint32_t mMipLevels;
    uint32_t mArraySize;
    uint32_t mSampleCount;

    VmaAllocation mAllocation;
    VmaAllocationInfo mAllocationInfo;

    vk::Image mHandle;

    Type_ImageViews mViews;
};

class ImageView {
public:
    ImageView(Device* device, vk::ImageSubresourceRange range,
              vk::Image imageHandle, vk::Format format, vk::ImageViewType type);

    ~ImageView();

    vk::ImageView GetHandle() const;

    void Destroy();

    friend class Texture;

private:
    vk::ImageView CreateImageView() const;

private:
    Device* pDevice;
    vk::ImageSubresourceRange mRange;
    vk::Image mImageHandle;
    vk::Format mFormat;
    vk::ImageViewType mType;

    vk::ImageView mHandle;

    Type_STLString mName {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core