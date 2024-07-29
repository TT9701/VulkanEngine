#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanPhysicalDevice;

class VulkanDevice {
public:
    VulkanDevice(VulkanPhysicalDevice*               physicalDevice,
                 ::std::vector<::std::string> const& requestedLayers     = {},
                 ::std::vector<::std::string> const& requestedExtensions = {},
                 vk::PhysicalDeviceFeatures*         pFeatures           = {},
                 void*                               pNext = nullptr);

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

private:
    vk::Device CreateDevice(std::vector<std::string> const& requestedLayers,
                            std::vector<std::string> const& requestedExtensions,
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