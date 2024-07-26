#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanPhysicalDevice;

class VulkanDevice {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanDevice(Type_SPInstance<VulkanPhysicalDevice> const& physicalDevice,
                 ::std::vector<::std::string> const& requestedLayers = {},
                 ::std::vector<::std::string> const& requestedExtensions = {},
                 vk::PhysicalDeviceFeatures* pFeatures = {},
                 void* pNext = nullptr);

    ~VulkanDevice();
    MOVABLE_ONLY(VulkanDevice);

public:
    vk::Device const& GetHandle() const { return mDevice; }

    ::std::vector<vk::Queue> const& GetGraphicQueues() const {
        return mGraphicQueues;
    }

    ::std::vector<vk::Queue> const& GetComputeQueues() const {
        return mComputeQueues;
    }

    ::std::vector<vk::Queue> const& GetTransferQueues() const {
        return mTransferQueues;
    }

private:
    vk::Device CreateDevice(std::vector<std::string> const& requestedLayers,
                            std::vector<std::string> const& requestedExtensions,
                            vk::PhysicalDeviceFeatures* pFeatures, void* pNext);

    void SetQueues();

private:
    Type_SPInstance<VulkanPhysicalDevice> pPhysicalDevice;

    ::std::vector<::std::string> enabledLayers {};
    ::std::vector<::std::string> enabledExtensions {};

    vk::Device mDevice;

    ::std::vector<vk::Queue> mGraphicQueues {};
    ::std::vector<vk::Queue> mComputeQueues {};
    ::std::vector<vk::Queue> mTransferQueues {};
};