#include "VulkanPhysicalDevice.hpp"

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/Logger.hpp"
#include "VulkanInstance.hpp"

VulkanPhysicalDevice::VulkanPhysicalDevice(VulkanInstance* instance,
                                           vk::QueueFlags requestedQueueTypes)
    : pInstance(instance),
      mPhysicalDevice(PickPhysicalDevice(requestedQueueTypes)) {
    DBG_LOG_INFO("Physical Device Selected: %s",
                 mPhysicalDevice.getProperties().deviceName.data());

    SetQueueFamlies(requestedQueueTypes);
}

vk::PhysicalDevice VulkanPhysicalDevice::PickPhysicalDevice(
    vk::QueueFlags requestedQueueTypes) {
    auto deviceList = pInstance->GetHandle().enumeratePhysicalDevices();
    VE_ASSERT(!deviceList.empty(), "device list is empty");

    vk::PhysicalDevice picked;

    for (auto& device : deviceList) {
        std::string devicename(device.getProperties().deviceName.data());
        const auto result = devicename.find("NVIDIA");
        if (result != std::string::npos) {
            picked = device;
            break;
        }
    }

    return picked;
}

void VulkanPhysicalDevice::SetQueueFamlies(vk::QueueFlags requestedQueueTypes) {
    auto queueFamilyProps = mPhysicalDevice.getQueueFamilyProperties();

    for (uint32_t queueFamilyIndex = 0;
         queueFamilyIndex < queueFamilyProps.size()
         && static_cast<uint32_t>(requestedQueueTypes) != 0;
         ++queueFamilyIndex) {
        if (!mGraphicsFamilyIndex.has_value()
            && (requestedQueueTypes
                & queueFamilyProps[queueFamilyIndex].queueFlags)
                   & vk::QueueFlagBits::eGraphics) {
            mGraphicsFamilyIndex = queueFamilyIndex;
            mGraphicsQueueCount = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eGraphics;
            continue;
        }

        if (!mComputeFamilyIndex.has_value()
            && (requestedQueueTypes
                & queueFamilyProps[queueFamilyIndex].queueFlags)
                   & vk::QueueFlagBits::eCompute) {
            mComputeFamilyIndex = queueFamilyIndex;
            mComputeQueueCount = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eCompute;
            continue;
        }

        if (!mTransferFamilyIndex.has_value()
            && (requestedQueueTypes
                & queueFamilyProps[queueFamilyIndex].queueFlags)
                   & vk::QueueFlagBits::eTransfer) {
            mTransferFamilyIndex = queueFamilyIndex;
            mTransferQueueCount = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eTransfer;
        }
    }
}