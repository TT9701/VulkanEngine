#include "VulkanBuffer.hpp"

AllocatedVulkanBuffer::AllocatedVulkanBuffer(VmaAllocator         allocator,
                                             size_t               allocByteSize,
                                             vk::BufferUsageFlags usage,
                                             VmaAllocationCreateFlags flags)
    : mAllocator(allocator) {
    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaAllocInfo.flags = flags;

    vmaCreateBuffer(allocator,
                    reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo),
                    &vmaAllocInfo, reinterpret_cast<VkBuffer*>(&mBuffer),
                    &mAllocation, &mInfo);
}

AllocatedVulkanBuffer::~AllocatedVulkanBuffer() {
    vmaDestroyBuffer(mAllocator, mBuffer, mAllocation);
}