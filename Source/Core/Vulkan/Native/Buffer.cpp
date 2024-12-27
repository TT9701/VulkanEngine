#include "Buffer.h"

#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Device.h"
#include "MemoryAllocator.h"

namespace IntelliDesign_NS::Vulkan::Core {

Buffer::Buffer(VulkanContext& context, size_t size, vk::BufferUsageFlags usage,
               MemoryType memType, size_t stride)
    : mContext(context),
      mUsageFlags(usage),
      mMemoryType(memType),
      mSize(size),
      mStride(stride),
      mHandle(CreateBufferResource()) {}

Buffer::~Buffer() {
    Destroy();
}

vk::Buffer Buffer::GetHandle() const {
    return mHandle;
}

vk::DeviceAddress Buffer::GetDeviceAddress() const {
    if (bDeviceAddressEnabled) {
        vk::BufferDeviceAddressInfo info {mHandle};
        return mContext.GetDevice()->getBufferAddress(info);
    }
    return 0;
}

Buffer::MemoryType Buffer::GetMemoryType() const {
    return mMemoryType;
}

void* Buffer::GetMapPtr() const {
    if (bMapped) {
        return mAllocationInfo.pMappedData;
    }
    return nullptr;
}

void Buffer::SetName(const char* name) const {
    mContext.GetDevice().SetObjectName(mHandle, name);
    if (mAllocationInfo.deviceMemory != VK_NULL_HANDLE)
        mContext.GetDevice().SetObjectName(
            vk::DeviceMemory(mAllocationInfo.deviceMemory), name);
}

void Buffer::Resize(size_t newSize) {
    Destroy();
    mSize = newSize;
    mAllocation = {};
    mAllocationInfo = {};
    mHandle = VK_NULL_HANDLE;

    mHandle = CreateBufferResource();
}

vk::Buffer Buffer::CreateBufferResource() {
    bMapped = mMemoryType != MemoryType::DeviceLocal;
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
        case MemoryType::Staging:
            vmaAllocInfo.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryType::DeviceLocal:
            vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            break;
        case MemoryType::ReadBack:
            vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                               | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    VkBuffer buffer {};
    VK_CHECK((vk::Result)vmaCreateBuffer(
        mContext.GetVmaAllocator().GetHandle(),
        reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &vmaAllocInfo,
        &buffer, &mAllocation, &mAllocationInfo));

    return buffer;
}

void Buffer::Destroy() {
    vmaDestroyBuffer(mContext.GetVmaAllocator().GetHandle(), mHandle,
                     mAllocation);
}

vk::BufferUsageFlags Buffer::GetUsageFlags() const {
    return mUsageFlags;
}

size_t Buffer::GetSize() const {
    return mSize;
}

size_t Buffer::GetStride() const {
    return mStride;
}

uint32_t Buffer::GetCount() const {
    return mSize / mStride;
}

}  // namespace IntelliDesign_NS::Vulkan::Core