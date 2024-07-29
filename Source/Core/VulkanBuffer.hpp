#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanMemoryAllocator;

class VulkanAllocatedBuffer {
public:
    VulkanAllocatedBuffer(VulkanMemoryAllocator* allocator,
                          size_t allocByteSize, vk::BufferUsageFlags usage,
                          VmaAllocationCreateFlags flags);
    ~VulkanAllocatedBuffer();
    MOVABLE_ONLY(VulkanAllocatedBuffer);

public:
    vk::Buffer GetHandle() const { return mBuffer; }

    VmaAllocation GetAllocation() const { return mAllocation; }

    VmaAllocationInfo const& GetAllocationInfo() const { return mInfo; }

private:
    vk::Buffer CreateBuffer(size_t allocByteSize, vk::BufferUsageFlags usage,
                            VmaAllocationCreateFlags flags);

private:
    VulkanMemoryAllocator* mAllocator;

    VmaAllocation     mAllocation {};
    VmaAllocationInfo mInfo {};

    vk::Buffer mBuffer;
};