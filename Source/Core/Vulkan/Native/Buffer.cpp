#include "Buffer.hpp"

#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Device.hpp"
#include "MemoryAllocator.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

Buffer::Buffer(Device* device, MemoryAllocator* allocator, size_t size,
               vk::BufferUsageFlags usage, MemoryType memType)
    : pDevice(device),
      pAllocator(allocator),
      mUsageFlags(usage),
      mMemoryType(memType),
      mSize(size),
      mHandle(CreateBufferResource()) {}

Buffer::~Buffer() {
    vmaDestroyBuffer(pAllocator->GetHandle(), mHandle, mAllocation);
}

vk::Buffer Buffer::GetHandle() const {
    return mHandle;
}

vk::DeviceAddress Buffer::GetDeviceAddress() const {
    if (bDeviceAddressEnabled) {
        vk::BufferDeviceAddressInfo info {mHandle};
        return pDevice->GetHandle().getBufferAddress(info);
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
    pDevice->SetObjectName(mHandle, name);
    pDevice->SetObjectName(vk::DeviceMemory(mAllocationInfo.deviceMemory),
                           name);
}

void Buffer::AllocateDescriptor(DescriptorManager* manager, uint32_t binding,
                                const char* descSetName,
                                vk::DescriptorType type) const {
    vk::DescriptorAddressInfoEXT bufferInfo {};
    bufferInfo.setAddress(GetDeviceAddress()).setRange(GetSize());

    manager->CreateBufferDescriptor(manager->GetDescriptorSet(descSetName),
                                    binding, type, &bufferInfo);
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
        pAllocator->GetHandle(),
        reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &vmaAllocInfo,
        &buffer, &mAllocation, &mAllocationInfo));

    return buffer;
}

vk::BufferUsageFlags Buffer::GetUsageFlags() const {
    return mUsageFlags;
}

size_t Buffer::GetSize() const {
    return mSize;
}

}  // namespace IntelliDesign_NS::Vulkan::Core