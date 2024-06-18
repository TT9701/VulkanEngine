#pragma once

#include <Core/Utilities/VulkanUtilities.hpp>

class AllocatedVulkanBuffer {
public:
    void CreateBuffer(VmaAllocator allocator, size_t allocByteSize,
                      vk::BufferUsageFlags     usage,
                      VmaAllocationCreateFlags flags);

    void CreateExternalBuffer(VmaAllocator allocator, size_t allocByteSize,
                              vk::BufferUsageFlags     usage,
                              VmaAllocationCreateFlags flags, VmaPool pool);

    void Destroy(VmaAllocator allocator);

    vk::Buffer        mBuffer {};
    VmaAllocation     mAllocation {};
    VmaAllocationInfo mInfo {};
};