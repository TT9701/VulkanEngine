#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;
class Device;
class Instance;

class MemoryAllocator {
public:
    MemoryAllocator(PhysicalDevice* physicalDevice,
                          Device* device, Instance* instance);

    ~MemoryAllocator();

    CLASS_MOVABLE_ONLY(MemoryAllocator);

public:
    VmaAllocator GetHandle() const { return mAllocator; }

private:
    VmaAllocator CreateAllocator();

private:
    PhysicalDevice* pPhysicalDevice;
    Device* pDevice;
    Instance* pInstance;

    VmaAllocator mAllocator;
};

class ExternalMemoryPool {
public:
    ExternalMemoryPool(MemoryAllocator* allocator);
    ~ExternalMemoryPool();
    CLASS_MOVABLE_ONLY(ExternalMemoryPool);

public:
    VmaPool GetHandle() const { return mPool; }

private:
    VmaPool CreatePool();

private:
    MemoryAllocator* pAllocator;

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};

    VmaPool mPool;
};

}  // namespace IntelliDesign_NS::Vulkan::Core