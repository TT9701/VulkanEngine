#include "MemoryAllocator.h"

#include "Core/Utilities/Logger.h"
#include "Device.h"
#include "Instance.h"
#include "PhysicalDevice.h"

namespace IntelliDesign_NS::Vulkan::Core {

MemoryAllocator::MemoryAllocator(PhysicalDevice& physicalDevice, Device& device,
                                 Instance& instance)
    : mPhysicalDevice(physicalDevice),
      mDevice(device),
      mInstance(instance),
      mHandle(CreateAllocator()) {
    DBG_LOG_INFO("vma Allocator Created");
}

MemoryAllocator::~MemoryAllocator() {
    if (mHandle != VK_NULL_HANDLE) {
        VmaTotalStatistics stats;
        vmaCalculateStatistics(mHandle, &stats);
        DBG_LOG_INFO("Total device memory leaked: %d bytes.",
                     stats.total.statistics.allocationBytes);
        vmaDestroyAllocator(mHandle);
        mHandle = VK_NULL_HANDLE;
    }
}

VmaAllocator MemoryAllocator::CreateAllocator() {
    const VmaVulkanFunctions vulkanFunctions = {
        .vkGetInstanceProcAddr =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
        .vkGetPhysicalDeviceProperties =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory,
        .vkFreeMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory,
        .vkMapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory,
        .vkUnmapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory,
        .vkFlushMappedMemoryRanges =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory,
        .vkBindImageMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory,
        .vkGetBufferMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements,
        .vkCreateBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer,
        .vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer,
        .vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage,
        .vkDestroyImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage,
        .vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer,
#if VMA_VULKAN_VERSION >= 1001000
        .vkGetBufferMemoryRequirements2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2,
        .vkBindBufferMemory2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2,
        .vkBindImageMemory2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2,
        .vkGetPhysicalDeviceMemoryProperties2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2,
#endif
#if VMA_VULKAN_VERSION >= 1003000
        .vkGetDeviceBufferMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceBufferMemoryRequirements,
        .vkGetDeviceImageMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements,
#endif
    };

    VmaAllocatorCreateInfo allocInfo = {
        .flags = 0,
        .physicalDevice = mPhysicalDevice.GetHandle(),
        .device = mDevice.GetHandle(),
        .pVulkanFunctions = &vulkanFunctions,
        .instance = mInstance.GetHandle(),
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };

    bool can_get_memory_requirements = mDevice.IsExtensionSupported(
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    bool has_dedicated_allocation = mDevice.IsExtensionSupported(
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    if (can_get_memory_requirements && has_dedicated_allocation
        && mDevice.IsExtensionEnabled(
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }

    if (mDevice.IsExtensionSupported(
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
        && mDevice.IsExtensionEnabled(
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }

    if (mDevice.IsExtensionSupported(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)
        && mDevice.IsExtensionEnabled(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }

    if (mDevice.IsExtensionSupported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)
        && mDevice.IsExtensionEnabled(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

    if (mDevice.IsExtensionSupported(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME)
        && mDevice.IsExtensionEnabled(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    }

    if (mDevice.IsExtensionSupported(
            VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)
        && mDevice.IsExtensionEnabled(
            VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    }

    VmaAllocator al {};
    vmaCreateAllocator(&allocInfo, &al);
    return al;
}

ExternalMemoryPool::ExternalMemoryPool(MemoryAllocator& allocator)
    : mAllocator(allocator), mPool(CreatePool()) {
    DBG_LOG_INFO("vma External Resource Pool Created");
}

ExternalMemoryPool::~ExternalMemoryPool() {
    vmaDestroyPool(mAllocator.GetHandle(), mPool);
}

VmaPool ExternalMemoryPool::CreatePool() {
    VmaPoolCreateInfo vmaPoolCreateInfo {};
    vmaPoolCreateInfo.pMemoryAllocateNext = &mExportMemoryAllocateInfo;

    VmaPool pool {};

    vmaCreatePool(mAllocator.GetHandle(), &vmaPoolCreateInfo, &pool);

    return pool;
}

AllocatedBase::AllocatedBase(MemoryAllocator& allocator,
                             const VmaAllocationCreateInfo& allocCreateInfo)
    : mAllocator(&allocator), mAllocCreateInfo(allocCreateInfo) {}

AllocatedBase::AllocatedBase(AllocatedBase&& other) noexcept
    : mAllocator(other.mAllocator),
      mAllocCreateInfo(std::exchange(other.mAllocCreateInfo, {})),
      mAllocation(std::exchange(other.mAllocation, {})),
      mMappedData(std::exchange(other.mMappedData, {})),
      mCoherent(std::exchange(other.mCoherent, {})),
      mPersistent(std::exchange(other.mPersistent, {})) {}

const uint8_t* AllocatedBase::GetData() const {
    return mMappedData;
}

VkDeviceMemory AllocatedBase::GetMemory() const {
    VmaAllocationInfo info;
    vmaGetAllocationInfo(mAllocator->GetHandle(), mAllocation, &info);
    return info.deviceMemory;
}

void AllocatedBase::Flush(VkDeviceSize offset, VkDeviceSize size) {
    if (!mCoherent) {
        vmaFlushAllocation(mAllocator->GetHandle(), mAllocation, offset, size);
    }
}

uint8_t* AllocatedBase::Map() {
    if (!mPersistent && !Mapped()) {
        VK_CHECK(
            (vk::Result)vmaMapMemory(mAllocator->GetHandle(), mAllocation,
                                     reinterpret_cast<void**>(&mMappedData)));
        assert(mMappedData);
    }
    return mMappedData;
}

void AllocatedBase::Unmap() {
    if (!mPersistent && Mapped()) {
        vmaUnmapMemory(mAllocator->GetHandle(), mAllocation);
        mMappedData = nullptr;
    }
}

size_t AllocatedBase::Update(const uint8_t* data, size_t size, size_t offset) {
    if (mPersistent) {
        std::copy(data, data + size, mMappedData + offset);
        Flush();
    } else {
        Map();
        std::copy(data, data + size, mMappedData + offset);
        Flush();
        Unmap();
    }
    return size;
}

size_t AllocatedBase::Update(void const* data, size_t size, size_t offset) {
    return Update(static_cast<const uint8_t*>(data), size, offset);
}

void AllocatedBase::PostCreate(VmaAllocationInfo const& allocationInfo) {
    VkMemoryPropertyFlags prop;
    vmaGetAllocationMemoryProperties(mAllocator->GetHandle(), mAllocation,
                                     &prop);
    mCoherent = (prop & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
             == VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mMappedData = static_cast<uint8_t*>(allocationInfo.pMappedData);
    mPersistent = Mapped();
}

[[nodiscard]] VkBuffer AllocatedBase::CreateBuffer(
    VkBufferCreateInfo const& createInfo) {
    VkBuffer handleResult = VK_NULL_HANDLE;
    VmaAllocationInfo info {};

    auto result =
        vmaCreateBuffer(mAllocator->GetHandle(), &createInfo, &mAllocCreateInfo,
                        &handleResult, &mAllocation, &info);

    if (result != VK_SUCCESS) {
        throw ::std::runtime_error {vk::to_string((vk::Result)result)
                                    + "Cannot create Buffer"};
    }
    PostCreate(info);
    return handleResult;
}

[[nodiscard]] VkImage AllocatedBase::CreateImage(
    VkImageCreateInfo const& createInfo) {
    assert(0 < createInfo.mipLevels && "Images should have at least one level");
    assert(0 < createInfo.arrayLayers
           && "Images should have at least one layer");
    assert(0 < createInfo.usage
           && "Images should have at least one usage type");

    VkImage handleResult = VK_NULL_HANDLE;
    VmaAllocationInfo info {};

    // If the image is an attachment, prefer dedicated memory
    constexpr VkImageUsageFlags attachmentOnlyFlags =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
    if (createInfo.usage & attachmentOnlyFlags) {
        mAllocCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    if (createInfo.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
        mAllocCreateInfo.preferredFlags |=
            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    }

    auto result =
        vmaCreateImage(mAllocator->GetHandle(), &createInfo, &mAllocCreateInfo,
                       &handleResult, &mAllocation, &info);

    if (result != VK_SUCCESS) {
        throw ::std::runtime_error {vk::to_string((vk::Result)result)
                                    + "Cannot create Image"};
    }

    PostCreate(info);
    return handleResult;
}

void AllocatedBase::DestroyBuffer(VkBuffer buffer) {
    if (buffer != VK_NULL_HANDLE && mAllocation != VK_NULL_HANDLE) {
        Unmap();
        vmaDestroyBuffer(mAllocator->GetHandle(), buffer, mAllocation);
        Clear();
    }
}

void AllocatedBase::DestroyImage(VkImage image) {
    if (image != VK_NULL_HANDLE && mAllocation != VK_NULL_HANDLE) {
        Unmap();
        vmaDestroyImage(mAllocator->GetHandle(), image, mAllocation);
        Clear();
    }
}

bool AllocatedBase::Mapped() const {
    return mMappedData != nullptr;
}

void AllocatedBase::Clear() {
    mMappedData = nullptr;
    mPersistent = false;
    mAllocCreateInfo = {};
}

}  // namespace IntelliDesign_NS::Vulkan::Core