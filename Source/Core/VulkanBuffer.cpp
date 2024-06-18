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
}

void AllocatedVulkanBuffer::CreateExternalBuffer(VmaAllocator allocator,
                                                 size_t       allocByteSize,
                                                 vk::BufferUsageFlags     usage,
                                                 VmaAllocationCreateFlags flags,
                                                 VmaPool pool) {

    vk::ExternalMemoryBufferCreateInfo externalbuffer {};
    externalbuffer.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);

    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage).setPNext(&externalbuffer);

    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaAllocInfo.flags = flags;
    vmaAllocInfo.pool  = pool;

    vmaCreateBuffer(allocator, (VkBufferCreateInfo*)&bufferInfo, &vmaAllocInfo,
                    (VkBuffer*)&mBuffer, &mAllocation, &mInfo);
}

void AllocatedVulkanBuffer::Destroy(VmaAllocator allocator) {
    vmaDestroyBuffer(allocator, mBuffer, mAllocation);
}