#include "Device.h"

#include "Core/Utilities/VulkanUtilities.h"
#include "PhysicalDevice.h"
#include "Surface.h"

namespace IntelliDesign_NS::Vulkan::Core {

Device::Device(PhysicalDevice& physicalDevice, Surface& surface,
               std::span<Type_STLString> requestedExtensions)
    : mPhysicalDevice(physicalDevice),
      mHandle(CreateDevice(requestedExtensions)) {
    DBG_LOG_INFO("Vulkan Device Created");

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mHandle);
#endif

    CreateQueues(surface);
}

Device::~Device() {
    mHandle.destroy();
}

vk::Device Device::GetHandle() const {
    return mHandle;
}

vk::Device const* Device::operator->() const {
    return &mHandle;
}

uint32_t Device::GetFamilyIndex(vk::QueueFlagBits queueFlag) const {
    const auto& queueFamilyPropertieses =
        mPhysicalDevice.GetQueueFamilyProperties();

    if (queueFlag & vk::QueueFlagBits::eCompute) {
        for (uint32_t i = 0;
             i < static_cast<uint32_t>(queueFamilyPropertieses.size()); i++) {
            if ((queueFamilyPropertieses[i].queueFlags & queueFlag)
                && !(queueFamilyPropertieses[i].queueFlags
                     & vk::QueueFlagBits::eGraphics)) {
                return i;
            }
        }
    }

    if (queueFlag & vk::QueueFlagBits::eTransfer) {
        for (uint32_t i = 0;
             i < static_cast<uint32_t>(queueFamilyPropertieses.size()); i++) {
            if ((queueFamilyPropertieses[i].queueFlags & queueFlag)
                && !(queueFamilyPropertieses[i].queueFlags
                     & vk::QueueFlagBits::eGraphics)
                && !(queueFamilyPropertieses[i].queueFlags
                     & vk::QueueFlagBits::eCompute)) {
                return i;
            }
        }
    }

    for (uint32_t i = 0;
         i < static_cast<uint32_t>(queueFamilyPropertieses.size()); i++) {
        if (queueFamilyPropertieses[i].queueFlags & queueFlag) {
            return i;
        }
    }

    throw std::runtime_error("Could not find a matching queue family index");
}

uint32_t Device::GetQueueFamilyIndex(vk::QueueFlagBits queueFlag) const {
    return mQueueFamilyIndices.at(queueFlag);
}

Type_STLUnorderedMap<vk::QueueFlagBits, uint32_t> const&
Device::GetQueueFamilyIndices() const {
    return mQueueFamilyIndices;
}

bool Device::IsExtensionSupported(const char* extension) const {
    return mPhysicalDevice.IsExtensionSupported(extension);
}

bool Device::IsExtensionEnabled(const char* extension) const {
    return std::ranges::find_if(
               enabledExtensions,
               [extension](Type_STLString const& enabled_extension) {
                   return enabled_extension == extension;
               })
        != enabledExtensions.end();
}

Queue const& Device::GetQueue(uint32_t familyIndex, uint32_t index) const {
    return mQueues[familyIndex][index];
}

vk::Device Device::CreateDevice(std::span<Type_STLString> requestedExtensions) {
    DBG_LOG_INFO("Selected GPU: %s",
                 mPhysicalDevice.GetProperties().deviceName.data());

    auto queueFamilyProperties = mPhysicalDevice.GetQueueFamilyProperties();
    ::std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos(
        queueFamilyProperties.size());
    ::std::vector<::std::vector<float>> queuePriorities(
        queueFamilyProperties.size());

    for (uint32_t queueFamilyIndex = 0U;
         queueFamilyIndex < queueFamilyProperties.size(); ++queueFamilyIndex) {
        vk::QueueFamilyProperties const& queueFamilyProperty =
            queueFamilyProperties[queueFamilyIndex];

        if (mPhysicalDevice.HasHighPriorityGraphicsQueue()) {
            uint32_t graphicsQueueFamily =
                GetFamilyIndex(vk::QueueFlagBits::eGraphics);
            if (graphicsQueueFamily == queueFamilyIndex) {
                queuePriorities[queueFamilyIndex].reserve(
                    queueFamilyProperty.queueCount);
                queuePriorities[queueFamilyIndex].push_back(1.0f);
                for (uint32_t i = 1; i < queueFamilyProperty.queueCount; i++) {
                    queuePriorities[queueFamilyIndex].push_back(0.5f);
                }
            } else {
                queuePriorities[queueFamilyIndex].resize(
                    queueFamilyProperty.queueCount, 0.5f);
            }
        } else {
            queuePriorities[queueFamilyIndex].resize(
                queueFamilyProperty.queueCount, 0.5f);
        }

        vk::DeviceQueueCreateInfo& queueCreateInfo =
            queueCreateInfos[queueFamilyIndex];

        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = queueFamilyProperty.queueCount;
        queueCreateInfo.pQueuePriorities =
            queuePriorities[queueFamilyIndex].data();
    }

    // Check extensions to enable Vma Dedicated Allocation
    bool canGetMemoryRequirements =
        IsExtensionSupported(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    bool hasDedicatedAllocation =
        IsExtensionSupported(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

    if (canGetMemoryRequirements && hasDedicatedAllocation) {
        enabledExtensions.emplace_back(
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        enabledExtensions.emplace_back(
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

        DBG_LOG_INFO("Dedicated Allocation enabled");
    }

    if (IsExtensionSupported(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME)) {
        auto hostQueryResetFeatures = mPhysicalDevice.GetExtensionFeatures<
            vk::PhysicalDeviceHostQueryResetFeatures>();

        if (hostQueryResetFeatures.hostQueryReset) {
            mPhysicalDevice
                .AddExtensionFeatures<
                    vk::PhysicalDeviceHostQueryResetFeatures>()
                .hostQueryReset = true;
            enabledExtensions.emplace_back(
                VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
            DBG_LOG_INFO("Host query reset enabled");
        }
    }

    // For performance queries, we also use host query reset since queryPool resets cannot
    // live in the same command buffer as beginQuery
    if (IsExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)
        && IsExtensionSupported(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME)) {
        auto performanceQueryFeaturesKhr = mPhysicalDevice.GetExtensionFeatures<
            vk::PhysicalDevicePerformanceQueryFeaturesKHR>();

        if (performanceQueryFeaturesKhr.performanceCounterQueryPools) {
            mPhysicalDevice
                .AddExtensionFeatures<
                    vk::PhysicalDevicePerformanceQueryFeaturesKHR>()
                .performanceCounterQueryPools = true;

            enabledExtensions.emplace_back(
                VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);

            DBG_LOG_INFO("Performance query enabled");
        }
    }

    // Check that extensions are supported before trying to create the device
    std::vector<Type_STLString> unsupportedExtensions {};
    for (auto& extension : requestedExtensions) {
        if (IsExtensionSupported(extension.c_str())) {
            enabledExtensions.emplace_back(extension);
        } else {
            unsupportedExtensions.emplace_back(extension);
        }
    }

    if (!enabledExtensions.empty()) {
        DBG_LOG_INFO("Device supports the following requested extensions:");
        for (auto& extension : enabledExtensions) {
            DBG_LOG_INFO("  \t%s", extension.c_str());
        }
    }

    if (!unsupportedExtensions.empty()) {
        auto error = false;
        for (auto& extension : unsupportedExtensions) {
            // auto extension_is_optional = requestedExtensions[extension];
            // if (extension_is_optional) {
            //     DBG_LOG_INFO(
            //         "Optional device extension %s not available, some features "
            //         "may be disabled",
            //         extension);
            // } else {
            DBG_LOG_INFO(
                "Required device extension %s not available, cannot run",
                extension.c_str());
            error = true;
            // }
        }

        if (error) {
            throw ::std::runtime_error(
                vk::to_string(vk::Result::eErrorExtensionNotPresent)
                + "Extensions not present");
        }
    }

    ::std::vector<const char*> extsCStr;
    extsCStr.reserve(enabledExtensions.size());
    for (auto& ext : enabledExtensions) {
        extsCStr.emplace_back(ext.c_str());
    }

    vk::DeviceCreateInfo createInfo(
        {}, queueCreateInfos, {}, extsCStr,
        &mPhysicalDevice.GetMutableRequestedFeatures());

    createInfo.pNext = mPhysicalDevice.GetExtensionFeatureChain();

    return mPhysicalDevice->createDevice(createInfo);
}

void Device::CreateQueues(Surface& surface) {
    auto queueFamilyProperties = mPhysicalDevice.GetQueueFamilyProperties();

    mQueues.resize(queueFamilyProperties.size());

    for (uint32_t queueFamilyIndex = 0;
         queueFamilyIndex < queueFamilyProperties.size(); ++queueFamilyIndex) {
        vk::QueueFamilyProperties const& queueFamilyProperty =
            queueFamilyProperties[queueFamilyIndex];

        vk::Bool32 presentSupported = mPhysicalDevice->getSurfaceSupportKHR(
            queueFamilyIndex, surface.GetHandle());

        for (uint32_t queueIndex = 0;
             queueIndex < queueFamilyProperty.queueCount; ++queueIndex) {
            mQueues[queueFamilyIndex].emplace_back(
                *this, queueFamilyIndex, queueFamilyProperty, presentSupported,
                queueIndex);

            auto& queue = mQueues[queueFamilyIndex].back();
            SetObjectName(
                queue.GetHandle(),
                vk::to_string(queue.GetFamilyProperties().queueFlags).c_str());
        }
    }

    mQueueFamilyIndices.emplace(vk::QueueFlagBits::eGraphics,
                                GetFamilyIndex(vk::QueueFlagBits::eGraphics));
    mQueueFamilyIndices.emplace(vk::QueueFlagBits::eCompute,
                                GetFamilyIndex(vk::QueueFlagBits::eCompute));
    mQueueFamilyIndices.emplace(vk::QueueFlagBits::eTransfer,
                                GetFamilyIndex(vk::QueueFlagBits::eTransfer));
}

}  // namespace IntelliDesign_NS::Vulkan::Core