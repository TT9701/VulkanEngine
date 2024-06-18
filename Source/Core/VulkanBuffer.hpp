#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

class AllocatedVulkanBuffer {
public:
    void CreateBuffer(VmaAllocator allocator, size_t allocByteSize,
                      vk::BufferUsageFlags     usage,
                      VmaAllocationCreateFlags flags);

    void Destroy();

    VmaAllocator      mAllocator {};
    vk::Buffer        mBuffer {};
    VmaAllocation     mAllocation {};
    VmaAllocationInfo mInfo {};
};