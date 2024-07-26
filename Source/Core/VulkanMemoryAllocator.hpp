#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanPhysicalDevice;
class VulkanDevice;
class VulkanInstance;

class VulkanMemoryAllocator {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanMemoryAllocator(
        Type_SPInstance<VulkanPhysicalDevice> const& physicalDevice,
        Type_SPInstance<VulkanDevice> const& device,
        Type_SPInstance<VulkanInstance> const& instance);

    ~VulkanMemoryAllocator();

    MOVABLE_ONLY(VulkanMemoryAllocator);

public:
    VmaAllocator const& GetHandle() const { return mAllocator; }

private:
    VmaAllocator CreateAllocator();

private:
    Type_SPInstance<VulkanPhysicalDevice> pPhysicalDevice;
    Type_SPInstance<VulkanDevice> pDevice;
    Type_SPInstance<VulkanInstance> pInstance;

    VmaAllocator mAllocator;
};

class VulkanExternalMemoryPool {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanExternalMemoryPool(
        Type_SPInstance<VulkanMemoryAllocator> const& allocator);
    ~VulkanExternalMemoryPool();
    MOVABLE_ONLY(VulkanExternalMemoryPool);

public:
    VmaPool const& GetHandle() const { return mPool; }

private:
    VmaPool CreatePool();

private:
    Type_SPInstance<VulkanMemoryAllocator> pAllocator;

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};

    VmaPool mPool;
};