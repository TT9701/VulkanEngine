#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Queue.h"

#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;
class Surface;

class Device {
public:
    Device(PhysicalDevice& physicalDevice, Surface& surface,
           ::std::span<Type_STLString> requestedExtensions = {});

    ~Device();
    CLASS_MOVABLE_ONLY(Device);

public:
    vk::Device GetHandle() const;

    vk::Device const* operator->() const;

    uint32_t GetQueueFamilyIndex(vk::QueueFlagBits queueFlag) const;

    bool IsExtensionSupported(const char* extension) const;

    bool IsExtensionEnabled(const char* extension) const;

    Queue const& GetQueue(uint32_t familyIndex, uint32_t index) const;

    template <class VkCppHandle>
    void SetObjectName(VkCppHandle handle, const char* name);

private:
    uint32_t GetFamilyIndex(vk::QueueFlagBits queueFlag) const;

    vk::Device CreateDevice(std::span<Type_STLString> requestedExtensions);

    void CreateQueues(Surface& surface);

private:
    PhysicalDevice& mPhysicalDevice;

    Type_STLVector<Type_STLString> enabledExtensions {};

    vk::Device mHandle;

    Type_STLVector<Type_STLVector<Queue>> mQueues {};
    Type_STLUnorderedMap<vk::QueueFlagBits, uint32_t> mQueueFamilyIndices;
};

}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::Vulkan::Core {

// TODO: template requirements
template <class VkCppHandle>
void Device::SetObjectName(VkCppHandle handle, const char* name) {
#ifndef NDEBUG
    vk::DebugUtilsObjectNameInfoEXT info {};
    using CType = typename VkCppHandle::CType;
    info.setObjectHandle((uint64_t)(CType)handle)
        .setObjectType(VkCppHandle::objectType)
        .setPObjectName(name);
    mHandle.setDebugUtilsObjectNameEXT(info);
#endif
}
}  // namespace IntelliDesign_NS::Vulkan::Core