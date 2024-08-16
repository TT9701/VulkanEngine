#include "VulkanBuffer.hpp"

#include "VulkanDevice.hpp"
#include "VulkanHelper.hpp"
#include "VulkanMemoryAllocator.hpp"

VulkanBuffer::VulkanBuffer(VulkanDevice* device,
                           VulkanMemoryAllocator* allocator, size_t size,
                           vk::BufferUsageFlags usage, BufferMemoryType memType)
    : pDevice(device),
      pAllocator(allocator),
      mUsageFlags(usage),
      mMemoryType(memType),
      mSize(size),
      mHandle(CreateBufferResource()) {}

VulkanBuffer::~VulkanBuffer() {
    vmaDestroyBuffer(pAllocator->GetHandle(), mHandle, mAllocation);
}

vk::Buffer VulkanBuffer::GetHandle() const {
    return mHandle;
}

vk::DeviceAddress VulkanBuffer::GetDeviceAddress() const {
    if (bDeviceAddressEnabled) {
        vk::BufferDeviceAddressInfo info {mHandle};
        return pDevice->GetHandle().getBufferAddress(info);
    }
    return 0;
}

BufferMemoryType VulkanBuffer::GetMemoryType() const {
    return mMemoryType;
}

void* VulkanBuffer::GetMapPtr() const {
    if (bMapped) {
        return mAllocationInfo.pMappedData;
    }
    return nullptr;
}

vk::Buffer VulkanBuffer::CreateBufferResource() {
    bMapped = mMemoryType != BufferMemoryType::DeviceLocal;
    if (mUsageFlags & vk::BufferUsageFlagBits::eShaderDeviceAddress
        || mUsageFlags & vk::BufferUsageFlagBits::eShaderDeviceAddressEXT
        || mUsageFlags & vk::BufferUsageFlagBits::eShaderDeviceAddressKHR) {
        bDeviceAddressEnabled = true;
    }

    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(mSize).setUsage(mUsageFlags);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    switch (mMemoryType) {
        case BufferMemoryType::Staging:
            vmaAllocInfo.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case BufferMemoryType::DeviceLocal:
            vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            break;
        case BufferMemoryType::ReadBack:
            vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                               | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    VkBuffer buffer {};
    VK_CHECK((vk::Result)vmaCreateBuffer(
        pAllocator->GetHandle(),
        reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &vmaAllocInfo,
        &buffer, &mAllocation, &mAllocationInfo));

    return buffer;
}

vk::BufferUsageFlags VulkanBuffer::GetUsageFlags() const {
    return mUsageFlags;
}

size_t VulkanBuffer::GetSize() const {
    return mSize;
}