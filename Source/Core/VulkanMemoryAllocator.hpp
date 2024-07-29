#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanPhysicalDevice;
class VulkanDevice;
class VulkanInstance;

class VulkanMemoryAllocator {
public:
    VulkanMemoryAllocator(VulkanPhysicalDevice* physicalDevice,
                          VulkanDevice* device, VulkanInstance* instance);

    ~VulkanMemoryAllocator();

    MOVABLE_ONLY(VulkanMemoryAllocator);

public:
    VmaAllocator GetHandle() const { return mAllocator; }

private:
    VmaAllocator CreateAllocator();

private:
    VulkanPhysicalDevice* pPhysicalDevice;
    VulkanDevice* pDevice;
    VulkanInstance* pInstance;

    VmaAllocator mAllocator;
};

class VulkanExternalMemoryPool {
public:
    VulkanExternalMemoryPool(VulkanMemoryAllocator* allocator);
    ~VulkanExternalMemoryPool();
    MOVABLE_ONLY(VulkanExternalMemoryPool);

public:
    VmaPool GetHandle() const { return mPool; }

private:
    VmaPool CreatePool();

private:
    VulkanMemoryAllocator* pAllocator;

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};

    VmaPool mPool;
};