#include "VulkanDevice.hpp"

#include "Utilities/Logger.hpp"
#include "Utilities/VulkanUtilities.hpp"
#include "VulkanPhysicalDevice.hpp"

VulkanDevice::VulkanDevice(
    Type_SPInstance<VulkanPhysicalDevice> const& physicalDevice,
    std::vector<std::string> const& requestedLayers,
    std::vector<std::string> const& requestedExtensions,
    vk::PhysicalDeviceFeatures* pFeatures, void* pNext)
    : pPhysicalDevice(physicalDevice),
      mDevice(CreateDevice(requestedLayers, requestedExtensions, pFeatures,
                           pNext)) {
    DBG_LOG_INFO("Vulkan Device Created");

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice);
#endif

    SetQueues();
}

VulkanDevice::~VulkanDevice() {
    mDevice.destroy();
}

vk::Device VulkanDevice::CreateDevice(
    std::vector<std::string> const& requestedLayers,
    std::vector<std::string> const& requestedExtensions,
    vk::PhysicalDeviceFeatures* pFeatures, void* pNext) {
    ::std::vector<float> queuePriorities(16, 1.0f);

    /**
     * TODO: Device layers & extensions
     */

    auto availableLayerProps =
        pPhysicalDevice->GetHandle().enumerateDeviceLayerProperties();
    ::std::vector<::std::string> availableLayers {};
    for (auto& prop : availableLayerProps) {
        availableLayers.push_back(prop.layerName);
    }
    enabledLayers = Utils::FilterStringList(availableLayers, requestedLayers);
    ::std::vector<const char*> enabledLayersCStr(enabledLayers.size());
    ::std::ranges::transform(enabledLayers, enabledLayersCStr.begin(),
                             ::std::mem_fn(&::std::string::c_str));

    auto availableExtensionProps =
        pPhysicalDevice->GetHandle().enumerateDeviceExtensionProperties();
    ::std::vector<::std::string> availableExtensions {};
    for (auto& prop : availableExtensionProps) {
        availableExtensions.push_back(prop.extensionName);
    }
    enabledExtensions =
        Utils::FilterStringList(availableExtensions, requestedExtensions);
    ::std::vector<const char*> enabledExtensionsCStr(enabledExtensions.size());
    ::std::ranges::transform(enabledExtensions, enabledExtensionsCStr.begin(),
                             ::std::mem_fn(&::std::string::c_str));

    ::std::vector<vk::DeviceQueueCreateInfo> queueCIs {};
    if (pPhysicalDevice->GetGraphicsQueueFamilyIndex().has_value())
        queueCIs.push_back(
            {{},
             pPhysicalDevice->GetGraphicsQueueFamilyIndex().value(),
             pPhysicalDevice->GetGraphicsQueueCount(),
             queuePriorities.data()});
    if (pPhysicalDevice->GetComputeQueueFamilyIndex().has_value())
        queueCIs.push_back(
            {{},
             pPhysicalDevice->GetComputeQueueFamilyIndex().value(),
             pPhysicalDevice->GetComputeQueueCount(),
             queuePriorities.data()});
    if (pPhysicalDevice->GetTransferQueueFamilyIndex().has_value())
        queueCIs.push_back(
            {{},
             pPhysicalDevice->GetTransferQueueFamilyIndex().value(),
             pPhysicalDevice->GetTransferQueueCount(),
             queuePriorities.data()});

    vk::DeviceCreateInfo deviceCreateInfo {};
    deviceCreateInfo.setQueueCreateInfos(queueCIs)
        .setPEnabledLayerNames(enabledLayersCStr)
        .setPEnabledExtensionNames(enabledExtensionsCStr)
        .setPEnabledFeatures(pFeatures)
        .setPNext(pNext);

    return pPhysicalDevice->GetHandle().createDevice(deviceCreateInfo);
}

void VulkanDevice::SetQueues() {
    mGraphicQueues.resize(pPhysicalDevice->GetGraphicsQueueCount());
    for (int i = 0; i < pPhysicalDevice->GetGraphicsQueueCount(); ++i)
        mGraphicQueues[i] = mDevice.getQueue(
            pPhysicalDevice->GetGraphicsQueueFamilyIndex().value(), i);

    mComputeQueues.resize(pPhysicalDevice->GetComputeQueueCount());
    for (int i = 0; i < pPhysicalDevice->GetComputeQueueCount(); ++i)
        mComputeQueues[i] = mDevice.getQueue(
            pPhysicalDevice->GetComputeQueueFamilyIndex().value(), i);

    mTransferQueues.resize(pPhysicalDevice->GetTransferQueueCount());
    for (int i = 0; i < pPhysicalDevice->GetTransferQueueCount(); ++i)
        mTransferQueues[i] = mDevice.getQueue(
            pPhysicalDevice->GetTransferQueueFamilyIndex().value(), i);
}