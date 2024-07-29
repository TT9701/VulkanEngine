#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanMemoryAllocator;

class VulkanAllocatedBuffer {
public:
    VulkanAllocatedBuffer(
        SharedPtr<VulkanMemoryAllocator> const& allocator,
        size_t allocByteSize, vk::BufferUsageFlags usage,
        VmaAllocationCreateFlags flags);
    ~VulkanAllocatedBuffer();
    MOVABLE_ONLY(VulkanAllocatedBuffer);

public:
    vk::Buffer const& GetHandle() const { return mBuffer; }

    VmaAllocation const& GetAllocation() const { return mAllocation; }

    VmaAllocationInfo const& GetAllocationInfo() const { return mInfo; }

private:
    vk::Buffer CreateBuffer(size_t allocByteSize, vk::BufferUsageFlags usage,
                            VmaAllocationCreateFlags flags);

private:
    SharedPtr<VulkanMemoryAllocator> mAllocator;

    VmaAllocation mAllocation {};
    VmaAllocationInfo mInfo {};

    vk::Buffer mBuffer;
};