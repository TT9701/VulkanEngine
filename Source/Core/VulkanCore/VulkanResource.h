#pragma once

#include "VulkanBuffer.hpp"
#include "VulkanTexture.hpp"

class VulkanResource {
public:
    enum class Type {
        Texture1D,
        Texture2D,
        Texture3D,
        TextureCube,
        Texture2DMultisample,
        Buffer
    };

    // buffer
    explicit VulkanResource(VulkanDevice* device,
                            VulkanMemoryAllocator* allocator, Type type,
                            size_t size, vk::BufferUsageFlags usage,
                            BufferMemoryType memType);

    // texture
    explicit VulkanResource(VulkanDevice* device,
                            VulkanMemoryAllocator* allocator, Type type,
                            vk::Format format, vk::Extent3D extent,
                            vk::ImageUsageFlags usage, uint32_t mipLevels,
                            uint32_t arraySize, uint32_t sampleCount);

    // texture for existing vk::image (swapchain images)
    explicit VulkanResource(VulkanDevice* device, vk::Image handle, Type type,
                            vk::Format format, vk::Extent3D extent,
                            uint32_t arraySize, uint32_t sampleCount);

    ~VulkanResource() = default;

public:
    Type GetType() const;
    ::std::string_view GetName() const;

    void SetName(::std::string const& name);

    // buffer
    vk::Buffer GetBufferHandle() const;
    void* GetBufferMappedPtr() const;
    vk::DeviceAddress GetBufferDeviceAddress() const;
    vk::BufferUsageFlags GetBufferUsageFlags() const;
    size_t GetBufferSize() const;
    BufferMemoryType GetBufferMemType() const;

    // texture
    void CreateTextureView(::std::string const& name,
                           vk::ImageAspectFlags aspect,
                           uint32_t mostDetailedMip = 0,
                           uint32_t mipCount = vk::RemainingMipLevels,
                           uint32_t firstArray = 0,
                           uint32_t arraySize = vk::RemainingArrayLayers);

    vk::Image GetTextureHandle() const;
    vk::ImageView GetTextureViewHandle(::std::string const& name) const;
    uint32_t GetWidth(uint32_t mipLevel = 0) const;
    uint32_t GetHeight(uint32_t mipLevel = 0) const;
    uint32_t GetDepth(uint32_t mipLevel = 0) const;
    uint32_t GetMipCount() const;
    uint32_t GetSampleCount() const;
    uint32_t GetArraySize() const;
    vk::Format GetFormat() const;

private:
    ::std::variant<VulkanBuffer, VulkanTexture> mResource;

    Type mType;
    ::std::string mName;
};
