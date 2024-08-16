#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanPhysicalDevice;

class VulkanDevice {
public:
    VulkanDevice(VulkanPhysicalDevice*       physicalDevice,
                 ::std::span<::std::string>  requestedLayers     = {},
                 ::std::span<::std::string>  requestedExtensions = {},
                 vk::PhysicalDeviceFeatures* pFeatures           = {},
                 void*                       pNext               = nullptr);

    ~VulkanDevice();
    MOVABLE_ONLY(VulkanDevice);

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
    vk::Device CreateDevice(std::span<std::string>      requestedLayers,
                            std::span<std::string>      requestedExtensions,
                            vk::PhysicalDeviceFeatures* pFeatures, void* pNext);

    void SetQueues();

private:
    VulkanPhysicalDevice* pPhysicalDevice;

    ::std::vector<::std::string> enabledLayers {};
    ::std::vector<::std::string> enabledExtensions {};

    vk::Device mDevice;

    ::std::vector<vk::Queue> mGraphicQueues {};
    ::std::vector<vk::Queue> mComputeQueues {};
    ::std::vector<vk::Queue> mTransferQueues {};
};

// TODO: template requirements
template <class VkCppHandle>
void VulkanDevice::SetObjectName(VkCppHandle handle, const char* name) {
    vk::DebugUtilsObjectNameInfoEXT info {};
    using CType = typename VkCppHandle::CType;
    info.setObjectHandle((uint64_t)(CType)handle)
        .setObjectType(VkCppHandle::objectType)
        .setPObjectName(name);
    mDevice.setDebugUtilsObjectNameEXT(info);
}