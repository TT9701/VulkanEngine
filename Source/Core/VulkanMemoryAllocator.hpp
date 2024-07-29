#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanPhysicalDevice;
class VulkanDevice;
class VulkanInstance;

class VulkanMemoryAllocator {
public:
    VulkanMemoryAllocator(
        SharedPtr<VulkanPhysicalDevice> const& physicalDevice,
        SharedPtr<VulkanDevice> const& device,
        SharedPtr<VulkanInstance> const& instance);

    ~VulkanMemoryAllocator();

    MOVABLE_ONLY(VulkanMemoryAllocator);

public:
    VmaAllocator const& GetHandle() const { return mAllocator; }

private:
    VmaAllocator CreateAllocator();

private:
    SharedPtr<VulkanPhysicalDevice> pPhysicalDevice;
    SharedPtr<VulkanDevice> pDevice;
    SharedPtr<VulkanInstance> pInstance;

    VmaAllocator mAllocator;
};

class VulkanExternalMemoryPool {
public:
    VulkanExternalMemoryPool(
        SharedPtr<VulkanMemoryAllocator> const& allocator);
    ~VulkanExternalMemoryPool();
    MOVABLE_ONLY(VulkanExternalMemoryPool);

public:
    VmaPool const& GetHandle() const { return mPool; }

private:
    VmaPool CreatePool();

private:
    SharedPtr<VulkanMemoryAllocator> pAllocator;

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};

    VmaPool mPool;
};