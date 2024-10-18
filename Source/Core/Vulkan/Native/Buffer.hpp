#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {

class Device;
class MemoryAllocator;
class DescriptorManager;

class Buffer {
public:
    enum class MemoryType { DeviceLocal, Staging, ReadBack };

    Buffer(Device* device, MemoryAllocator* allocator, size_t size,
           vk::BufferUsageFlags usage, MemoryType memType,
           size_t texelSize = 1);

    ~Buffer();

public:
    vk::Buffer GetHandle() const;

    // for buffers include vk::BufferUsageFlagBits::eShaderDeviceAddress flagbits
    // else return 0
    vk::DeviceAddress GetDeviceAddress() const;

    vk::BufferUsageFlags GetUsageFlags() const;

    size_t GetSize() const;

    size_t GetTexelSize() const;

    MemoryType GetMemoryType() const;

    // for mapped buffers, non mapped buffers return nullptr
    void* GetMapPtr() const;

    void SetName(const char* name) const;

    void Resize(size_t newSize);

private:
    vk::Buffer CreateBufferResource();

    void Destroy();

private:
    Device* pDevice;
    MemoryAllocator* pAllocator;

    vk::BufferUsageFlags mUsageFlags;
    MemoryType mMemoryType;

    size_t mSize;
    size_t mTexelSize;

    bool bMapped {false};
    bool bDeviceAddressEnabled {false};

    VmaAllocation mAllocation {};
    VmaAllocationInfo mAllocationInfo {};

    vk::Buffer mHandle;
};

}  // namespace IntelliDesign_NS::Vulkan::Core