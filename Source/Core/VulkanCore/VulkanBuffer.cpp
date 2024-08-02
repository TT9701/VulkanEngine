#include "VulkanBuffer.hpp"

#include "VulkanMemoryAllocator.hpp"

VulkanBuffer::VulkanBuffer(VulkanMemoryAllocator* allocator,
                           size_t allocByteSize, vk::BufferUsageFlags usage,
                           VmaAllocationCreateFlags flags,
                           VmaMemoryUsage           memoryUsage)
    : mAllocator(allocator),
      mBuffer(CreateBuffer(allocByteSize, usage, flags, memoryUsage)) {}

VulkanBuffer::~VulkanBuffer() {
    vmaDestroyBuffer(mAllocator->GetHandle(), mBuffer, mAllocation);
}

vk::Buffer VulkanBuffer::CreateBuffer(size_t                   allocByteSize,
                                      vk::BufferUsageFlags     usage,
                                      VmaAllocationCreateFlags flags,
                                      VmaMemoryUsage           memoryUsage) {
    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = flags;

    VkBuffer buffer {};

    vmaCreateBuffer(mAllocator->GetHandle(),
                    reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo),
                    &vmaAllocInfo, &buffer, &mAllocation, &mInfo);

    return buffer;
}