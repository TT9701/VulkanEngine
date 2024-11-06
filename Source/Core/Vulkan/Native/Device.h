#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;

class Device {
public:
    Device(PhysicalDevice* physicalDevice,
           ::std::span<Type_STLString> requestedLayers = {},
           ::std::span<Type_STLString> requestedExtensions = {},
           vk::PhysicalDeviceFeatures* pFeatures = {}, void* pNext = nullptr);

    ~Device();
    MOVABLE_ONLY(Device);

public:
    vk::Device GetHandle() const { return mDevice; }

    vk::Queue GetGraphicQueue(uint32_t index = 0) const {
        return mGraphicQueues[index];
    }

    vk::Queue GetComputeQueue(uint32_t index = 0) const {
        return mComputeQueues[index];
    }

    vk::Queue GetTransferQueue(uint32_t index = 0) const {
        return mTransferQueues[index];
    }

    template <class VkCppHandle>
    void SetObjectName(VkCppHandle handle, const char* name);

private:
    vk::Device CreateDevice(std::span<Type_STLString> requestedLayers,
                            std::span<Type_STLString> requestedExtensions,
                            vk::PhysicalDeviceFeatures* pFeatures, void* pNext);

    void SetQueues();

private:
    PhysicalDevice* pPhysicalDevice;

    Type_STLVector<Type_STLString> enabledLayers {};
    Type_STLVector<Type_STLString> enabledExtensions {};

    vk::Device mDevice;

    Type_STLVector<vk::Queue> mGraphicQueues {};
    Type_STLVector<vk::Queue> mComputeQueues {};
    Type_STLVector<vk::Queue> mTransferQueues {};
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
    mDevice.setDebugUtilsObjectNameEXT(info);
#endif
}
}  // namespace IntelliDesign_NS::Vulkan::Core