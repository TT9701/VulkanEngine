#include "VulkanBuffer.hpp"

void AllocatedVulkanBuffer::CreateBuffer(VmaAllocator             allocator,
                                         size_t                   allocByteSize,
                                         vk::BufferUsageFlags     usage,
                                         VmaAllocationCreateFlags flags) {

    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage);

    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaAllocInfo.flags = flags;

    vmaCreateBuffer(allocator, (VkBufferCreateInfo*)&bufferInfo, &vmaAllocInfo,
                    (VkBuffer*)&mBuffer, &mAllocation, &mInfo);

    mAllocator = allocator;
}

void AllocatedVulkanBuffer::Destroy() {
    vmaDestroyBuffer(mAllocator, mBuffer, mAllocation);
}