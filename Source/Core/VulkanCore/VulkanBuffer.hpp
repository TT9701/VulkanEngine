#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

class VulkanDevice;
class VulkanMemoryAllocator;

class VulkanBuffer {
public:
    enum class MemoryType { DeviceLocal, Staging, ReadBack };

    VulkanBuffer(VulkanDevice* device, VulkanMemoryAllocator* allocator,
                 size_t size, vk::BufferUsageFlags usage, MemoryType memType);

    ~VulkanBuffer();

public:
    vk::Buffer GetHandle() const;

    // for buffers include vk::BufferUsageFlagBits::eShaderDeviceAddress flagbits
    // else return 0
    vk::DeviceAddress GetDeviceAddress() const;

    vk::BufferUsageFlags GetUsageFlags() const;

    size_t GetSize() const;

    MemoryType GetMemoryType() const;

    // for mapped buffers, non mapped buffers return nullptr
    void* GetMapPtr() const;

private:
    vk::Buffer CreateBufferResource();

private:
    VulkanDevice* pDevice;
    VulkanMemoryAllocator* pAllocator;

    vk::BufferUsageFlags mUsageFlags;
    MemoryType mMemoryType;

    size_t mSize;

    bool bMapped {false};
    bool bDeviceAddressEnabled {false};

    VmaAllocation mAllocation {};
    VmaAllocationInfo mAllocationInfo {};

    vk::Buffer mHandle;
};