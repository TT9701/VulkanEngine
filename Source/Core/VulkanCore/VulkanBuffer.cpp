#include "VulkanBuffer.hpp"

#include "VulkanMemoryAllocator.hpp"

VulkanBuffer::VulkanBuffer(VulkanMemoryAllocator* allocator,
                           size_t allocByteSize, vk::BufferUsageFlags usage,
                           VmaAllocationCreateFlags flags)
    : mAllocator(allocator),
      mBuffer(CreateBuffer(allocByteSize, usage, flags)) {}

VulkanBuffer::~VulkanBuffer() {
    vmaDestroyBuffer(mAllocator->GetHandle(), mBuffer, mAllocation);
}

vk::Buffer VulkanBuffer::CreateBuffer(size_t                   allocByteSize,
                                      vk::BufferUsageFlags     usage,
                                      VmaAllocationCreateFlags flags) {
    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaAllocInfo.flags = flags;

    VkBuffer buffer {};

    vmaCreateBuffer(mAllocator->GetHandle(),
                    reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo),
                    &vmaAllocInfo, &buffer, &mAllocation, &mInfo);

    return buffer;
}