#include "PhysicalDevice.h"

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/Logger.h"
#include "Instance.h"

namespace IntelliDesign_NS::Vulkan::Core {

PhysicalDevice::PhysicalDevice(Instance& instance, vk::PhysicalDevice gpu)
    : VulkanResource(gpu), mInstance(instance) {
    auto& handle = GetHandle();
    mFeatures = handle.getFeatures();
    mProperties = handle.getProperties();
    mMemoryProperties = handle.getMemoryProperties();

    DBG_LOG_INFO(::std::string {"Found GPU: "} + mProperties.deviceName.data());

    auto queueFamilyProps = handle.getQueueFamilyProperties();
    mQueueFamilyProperties = Type_STLVector<vk::QueueFamilyProperties> {
        queueFamilyProps.begin(), queueFamilyProps.end()};

    auto deviceExt = handle.enumerateDeviceExtensionProperties();
    mDeviceExts = Type_STLVector<vk::ExtensionProperties> {deviceExt.begin(),
                                                           deviceExt.end()};

    // Display supported extensions
    if (!mDeviceExts.empty()) {
        DBG_LOG_INFO("HPPDevice supports the following extensions:");
        for (auto& extension : mDeviceExts) {
            DBG_LOG_INFO(::std::string {"  \t"}
                         + extension.extensionName.data());
        }
    }
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

void* PhysicalDevice::GetExtensionFeatureChain() const {
    return mLastRequestedExtensionFeature;
}

bool PhysicalDevice::IsExtensionSupported(
    ::std::string const& requested_extension) const {
    return std::ranges::find_if(mDeviceExts,
                                [requested_extension](auto& device_extension) {
                                    return std::strcmp(
                                               device_extension.extensionName,
                                               requested_extension.c_str())
                                        == 0;
                                })
        != mDeviceExts.end();
}

const vk::PhysicalDeviceFeatures& PhysicalDevice::GetFeatures() const {
    return mFeatures;
}

Instance& PhysicalDevice::GetInstance() const {
    return mInstance;
}

const vk::PhysicalDeviceMemoryProperties& PhysicalDevice::GetMemoryProperties()
    const {
    return mMemoryProperties;
}

uint32_t PhysicalDevice::GetMemoryType(uint32_t bits,
                                       vk::MemoryPropertyFlags properties,
                                       vk::Bool32* memory_type_found) const {
    for (uint32_t i = 0; i < mMemoryProperties.memoryTypeCount; i++) {
        if ((bits & 1) == 1) {
            if ((mMemoryProperties.memoryTypes[i].propertyFlags & properties)
                == properties) {
                if (memory_type_found) {
                    *memory_type_found = true;
                }
                return i;
            }
        }
        bits >>= 1;
    }

    if (memory_type_found) {
        *memory_type_found = false;
        return ~0;
    } else {
        throw std::runtime_error("Could not find a matching memory type");
    }
}

const vk::PhysicalDeviceProperties& PhysicalDevice::GetProperties() const {
    return mProperties;
}

const Type_STLVector<vk::QueueFamilyProperties>&
PhysicalDevice::GetQueueFamilyProperties() const {
    return mQueueFamilyProperties;
}

const vk::PhysicalDeviceFeatures PhysicalDevice::GetRequestedFeatures() const {
    return mRequestedFeatures;
}

vk::PhysicalDeviceFeatures& PhysicalDevice::GetMutableRequestedFeatures() {
    return mRequestedFeatures;
}

void PhysicalDevice::SetHighPriorityGraphicsQueueEnable(bool enable) {
    mHighPriorityGraphicsQueue = enable;
}

bool PhysicalDevice::HasHighPriorityGraphicsQueue() const {
    return mHighPriorityGraphicsQueue;
}

}  // namespace IntelliDesign_NS::Vulkan::Core