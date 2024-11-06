#include "MemoryAllocator.h"

#include "Core/Utilities/Logger.h"
#include "Device.h"
#include "Instance.h"
#include "PhysicalDevice.h"

namespace IntelliDesign_NS::Vulkan::Core {

MemoryAllocator::MemoryAllocator(
    PhysicalDevice* physicalDevice, Device* device,
    Instance* instance)
    : pPhysicalDevice(physicalDevice),
      pDevice(device),
      pInstance(instance),
      mAllocator(CreateAllocator()) {
    DBG_LOG_INFO("vma Allocator Created");
}

MemoryAllocator::~MemoryAllocator() {
    vmaDestroyAllocator(mAllocator);
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
#if defined(VK_KHR_buffer_device_address) && defined(_WIN32)
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
#endif
        .physicalDevice = pPhysicalDevice->GetHandle(),
        .device = pDevice->GetHandle(),
        .pVulkanFunctions = &vulkanFunctions,
        .instance = pInstance->GetHandle(),
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };

    VmaAllocator al {};
    vmaCreateAllocator(&allocInfo, &al);
    return al;
}

ExternalMemoryPool::ExternalMemoryPool(
    MemoryAllocator* allocator)
    : pAllocator(allocator), mPool(CreatePool()) {
    DBG_LOG_INFO("vma External Resource Pool Created");
}

ExternalMemoryPool::~ExternalMemoryPool() {
    vmaDestroyPool(pAllocator->GetHandle(), mPool);
}

VmaPool ExternalMemoryPool::CreatePool() {
    VmaPoolCreateInfo vmaPoolCreateInfo {};
    vmaPoolCreateInfo.pMemoryAllocateNext = &mExportMemoryAllocateInfo;

    VmaPool pool {};

    vmaCreatePool(pAllocator->GetHandle(), &vmaPoolCreateInfo, &pool);

    return pool;
}

}  // namespace IntelliDesign_NS::Vulkan::Core