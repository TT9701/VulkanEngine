#include "Device.hpp"

#include "Core/Utilities/VulkanUtilities.hpp"
#include "PhysicalDevice.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

Device::Device(PhysicalDevice* physicalDevice,
               std::span<Type_STLString> requestedLayers,
               std::span<Type_STLString> requestedExtensions,
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

Device::~Device() {
    mDevice.destroy();
}

vk::Device Device::CreateDevice(std::span<Type_STLString> requestedLayers,
                                std::span<Type_STLString> requestedExtensions,
                                vk::PhysicalDeviceFeatures* pFeatures,
                                void* pNext) {
    Type_STLVector<float> queuePriorities(16, 1.0f);

    /**
     * TODO: Device layers & extensions
     */

    auto availableLayerProps =
        pPhysicalDevice->GetHandle().enumerateDeviceLayerProperties();
    Type_STLVector<Type_STLString> availableLayers {};
    for (auto& prop : availableLayerProps) {
        availableLayers.push_back(prop.layerName.data());
    }
    enabledLayers = Utils::FilterStringList(availableLayers, requestedLayers);
    Type_STLVector<const char*> enabledLayersCStr(enabledLayers.size());
    ::std::ranges::transform(enabledLayers, enabledLayersCStr.begin(),
                             ::std::mem_fn(&Type_STLString::c_str));

    auto availableExtensionProps =
        pPhysicalDevice->GetHandle().enumerateDeviceExtensionProperties();
    Type_STLVector<Type_STLString> availableExtensions {};
    for (auto& prop : availableExtensionProps) {
        availableExtensions.push_back(prop.extensionName.data());
    }
    enabledExtensions =
        Utils::FilterStringList(availableExtensions, requestedExtensions);
    Type_STLVector<const char*> enabledExtensionsCStr(enabledExtensions.size());
    ::std::ranges::transform(enabledExtensions, enabledExtensionsCStr.begin(),
                             ::std::mem_fn(&Type_STLString::c_str));

    Type_STLVector<vk::DeviceQueueCreateInfo> queueCIs {};
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

void Device::SetQueues() {
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

}  // namespace IntelliDesign_NS::Vulkan::Core