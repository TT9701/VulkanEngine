#include "PhysicalDevice.h"

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/Logger.h"
#include "Instance.h"

namespace IntelliDesign_NS::Vulkan::Core {

PhysicalDevice::PhysicalDevice(Instance& instance,
                               vk::PhysicalDevice physicalDevice)
    : mInstance(instance), mHandle(physicalDevice) {
    mFeatures = physicalDevice.getFeatures2();
    mProperties = physicalDevice.getProperties2();
    mMemoryProperties = physicalDevice.getMemoryProperties2();

    DBG_LOG_INFO("Found GPU: %s", mProperties.properties.deviceName.data());

    auto queueProps = physicalDevice.getQueueFamilyProperties();
    mQueueFamilyProperties = Type_STLVector<vk::QueueFamilyProperties> {
        queueProps.begin(), queueProps.end()};

    auto ext = physicalDevice.enumerateDeviceExtensionProperties();
    mDeviceExtensions =
        Type_STLVector<vk::ExtensionProperties> {ext.begin(), ext.end()};

    SetQueueFamlies(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute
                    | vk::QueueFlagBits::eTransfer);
}

vk::PhysicalDeviceProperties const& PhysicalDevice::GetProperties() const {
    return mProperties.properties;
}

vk::PhysicalDeviceFeatures const& PhysicalDevice::GetFeatures() const {
    return mFeatures.features;
}

vk::PhysicalDeviceMemoryProperties const& PhysicalDevice::GetMemoryProperties()
    const {
    return mMemoryProperties.memoryProperties;
}

std::span<vk::QueueFamilyProperties>
PhysicalDevice::GetQueueFamilyProperties() {
    return mQueueFamilyProperties;
}

uint32_t PhysicalDevice::GetMemoryType(uint32_t bits,
                                       vk::MemoryPropertyFlags properties,
                                       vk::Bool32* memoryTypeFound) const {
    for (uint32_t i = 0; i < mMemoryProperties.memoryProperties.memoryTypeCount;
         i++) {
        if ((bits & 1) == 1) {
            if ((mMemoryProperties.memoryProperties.memoryTypes[i].propertyFlags
                 & properties)
                == properties) {
                if (memoryTypeFound) {
                    *memoryTypeFound = true;
                }
                return i;
            }
        }
        bits >>= 1;
    }

    if (memoryTypeFound) {
        *memoryTypeFound = false;
        return ~0;
    } else {
        throw std::runtime_error("Could not find a matching memory type");
    }
}

void* PhysicalDevice::GetExtensionFeatureChain() const {
    return mLastRequestedExtensionFeature;
}

vk::PhysicalDevice const& PhysicalDevice::GetHandle() const {
    return mHandle;
}

bool PhysicalDevice::IsExtensionSupported(const char* requestedExt) const {
    return std::ranges::find_if(mDeviceExtensions,
                                [requestedExt](auto& device_extension) {
                                    return std::strcmp(
                                               device_extension.extensionName,
                                               requestedExt)
                                        == 0;
                                })
        != mDeviceExtensions.end();
}

void PhysicalDevice::SetHighPriorityGraphicsQueue(bool enable) {
    mHighPriorityGraphicsQueue = enable;
}

bool PhysicalDevice::HasHighPriorityGraphicsQueue() const {
    return mHighPriorityGraphicsQueue;
}

vk::PhysicalDeviceFeatures const& PhysicalDevice::GetRequestedFeatures() const {
    return mRequestedFeatures.features;
}

vk::PhysicalDeviceFeatures& PhysicalDevice::GetMutableRequestedFeatures() {
    return mRequestedFeatures.features;
}

vk::PhysicalDevice const* PhysicalDevice::operator->() const {
    return &mHandle;
}

DriverVersion PhysicalDevice::GetDriverVersion() const {
    DriverVersion version;

    vk::PhysicalDeviceProperties const& properties = GetProperties();
    switch (properties.vendorID) {
        case 0x10DE:
            // Nvidia
            version.major = (properties.driverVersion >> 22) & 0x3ff;
            version.minor = (properties.driverVersion >> 14) & 0x0ff;
            version.patch = (properties.driverVersion >> 6) & 0x0ff;
            // Ignoring optional tertiary info in lower 6 bits
            break;
        case 0x8086:
            version.major = (properties.driverVersion >> 14) & 0x3ffff;
            version.minor = properties.driverVersion & 0x3ffff;
            break;
        default:
            version.major = VK_VERSION_MAJOR(properties.driverVersion);
            version.minor = VK_VERSION_MINOR(properties.driverVersion);
            version.patch = VK_VERSION_PATCH(properties.driverVersion);
            break;
    }

    return version;
}

void PhysicalDevice::SetQueueFamlies(vk::QueueFlags requestedQueueTypes) {
    auto queueFamilyProps = mHandle.getQueueFamilyProperties();

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

}  // namespace IntelliDesign_NS::Vulkan::Core