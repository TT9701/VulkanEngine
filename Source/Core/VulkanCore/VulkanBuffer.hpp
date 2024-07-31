#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanMemoryAllocator;

class VulkanBuffer {
public:
    VulkanBuffer(VulkanMemoryAllocator* allocator, size_t allocByteSize,
                 vk::BufferUsageFlags usage, VmaAllocationCreateFlags flags);
    ~VulkanBuffer();
    MOVABLE_ONLY(VulkanBuffer);

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