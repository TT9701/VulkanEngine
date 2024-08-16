#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {

class Device;
class MemoryAllocator;

class Texture {
public:
    enum class Type {
        Texture1D,
        Texture2D,
        Texture3D,
        TextureCube,
        Texture2DMultisample,
    };

    // No data source
    Texture(Device* device, MemoryAllocator* allocator,
                  Type type, vk::Format format, vk::Extent3D extent,
                  vk::ImageUsageFlags usage, uint32_t mipLevels,
                  uint32_t arraySize, uint32_t sampleCount);

    // from swapchain
    Texture(Device* device, vk::Image handle, Type type,
                  vk::Format format, vk::Extent3D extent, uint32_t arraySize,
                  uint32_t sampleCount);

    ~Texture();

public:
    void CreateImageView(::std::string_view name, vk::ImageAspectFlags aspect,
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
    vk::ImageView GetViewHandle(::std::string_view name) const;

private:
    vk::Image CreateImage();

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

    ::std::unordered_map<::std::string, vk::ImageView> mViews;
};

}  // namespace IntelliDesign_NS::Vulkan::Core