#pragma once

#include <variant>

#include "Buffer.h"
#include "Texture.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Sampler;
class DescriptorManager;
class VulkanContext;

class RenderResource {
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
    RenderResource(VulkanContext& context, Type type, size_t size,
                   vk::BufferUsageFlags usage, Buffer::MemoryType memType,
                   size_t texelSize = 1);

    // texture
    RenderResource(VulkanContext& context, Type type, vk::Format format,
                   vk::Extent3D extent, vk::ImageUsageFlags usage,
                   uint32_t mipLevels, uint32_t arraySize,
                   uint32_t sampleCount);

    // texture for existing vk::image (swapchain images)
    RenderResource(VulkanContext& context, vk::Image handle, Type type,
                   vk::Format format, vk::Extent3D extent, uint32_t arraySize,
                   uint32_t sampleCount);

    RenderResource(Buffer&& buffer);
    RenderResource(Texture&& texture);

    ~RenderResource() = default;

public:
    Type GetType() const;
    const char* GetName() const;

    void SetName(const char* name);

    // buffer
    vk::Buffer GetBufferHandle() const;
    void* GetBufferMappedPtr() const;
    vk::DeviceAddress GetBufferDeviceAddress() const;
    vk::BufferUsageFlags GetBufferUsageFlags() const;
    size_t GetBufferSize() const;
    Buffer::MemoryType GetBufferMemType() const;
    uint32_t GetBufferStride() const;
    void SetBufferDGCSequence(SharedPtr<DGCSequenceBase> const& dgcSeq);
    void Execute(vk::CommandBuffer cmd) const;

    // texture
    void CreateTexView(const char* name, vk::ImageAspectFlags aspect,
                       uint32_t mostDetailedMip = 0,
                       uint32_t mipCount = vk::RemainingMipLevels,
                       uint32_t firstArray = 0,
                       uint32_t arraySize = vk::RemainingArrayLayers);

    Texture const* GetTexPtr() const;
    vk::Image GetTexHandle() const;
    ImageView* GetTexView(const char* name = nullptr) const;
    vk::ImageView GetTexViewHandle(const char* name = nullptr) const;
    uint32_t GetTexWidth(uint32_t mipLevel = 0) const;
    uint32_t GetTexHeight(uint32_t mipLevel = 0) const;
    uint32_t GetTexDepth(uint32_t mipLevel = 0) const;
    uint32_t GetTexMipCount() const;
    uint32_t GetTexSampleCount() const;
    uint32_t GetTexArraySize() const;
    vk::ImageUsageFlags GetTexUsage() const;
    vk::Format GetTexFormat() const;

    void Resize(vk::Extent2D extent);

private:
    ::std::variant<Buffer, Texture> mResource;

    Type mType;
    Type_STLString mName;
};

}  // namespace IntelliDesign_NS::Vulkan::Core
