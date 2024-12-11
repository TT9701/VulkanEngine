#include "Instance.h"

#include "Core/Utilities/VulkanUtilities.h"

#include "PhysicalDevice.h"

namespace {
bool LoadDispatcher = false;

using namespace IntelliDesign_NS::Core::MemoryPool;

auto SetInstanceLayers(::std::span<Type_STLString> requestedLayers) {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    auto instanceLayersProps = vk::enumerateInstanceLayerProperties();
    Type_STLVector<Type_STLString> availableInstanceLayers {};
    for (auto& prop : instanceLayersProps) {
        availableInstanceLayers.push_back(prop.layerName.data());
    }
    return IntelliDesign_NS::Vulkan::Core::Utils::FilterStringList(
        availableInstanceLayers, requestedLayers);
}

auto SetInstanceExtensions(std::span<Type_STLString> requestedExtensions) {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    auto instanceExtensionProps = vk::enumerateInstanceExtensionProperties();
    Type_STLVector<Type_STLString> availableInstanceExtensions {};
    for (auto& prop : instanceExtensionProps) {
        availableInstanceExtensions.push_back(prop.extensionName.data());
    }
    return IntelliDesign_NS::Vulkan::Core::Utils::FilterStringList(
        availableInstanceExtensions, requestedExtensions);
}

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

Instance::Instance(std::span<Type_STLString> requestedInstanceLayers,
                   std::span<Type_STLString> requestedInstanceExtensions)
    : mEnabledLayers(SetInstanceLayers(requestedInstanceLayers)),
      mEnabledExtensions(SetInstanceExtensions(requestedInstanceExtensions)){
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    Type_STLVector<const char*> enabledLayersCStr(mEnabledLayers.size());
    for (int i = 0; i < mEnabledLayers.size(); ++i) {
        enabledLayersCStr[i] = mEnabledLayers[i].c_str();
    }

    Type_STLVector<const char*> enabledExtensionsCStr(
        mEnabledExtensions.size());
    for (int i = 0; i < mEnabledExtensions.size(); ++i) {
        enabledExtensionsCStr[i] = mEnabledExtensions[i].c_str();
    }

    vk::ApplicationInfo appInfo {};
    appInfo.setPEngineName("Vulkan Engine")
        .setPApplicationName("Fun")
        .setEngineVersion(1u)
        .setApplicationVersion(1u)
        .setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceCreateInfo {};
    instanceCreateInfo.setPApplicationInfo(&appInfo)
        .setPEnabledLayerNames(enabledLayersCStr)
        .setPEnabledExtensionNames(enabledExtensionsCStr);

    auto inst = vk::createInstance(instanceCreateInfo);

    mHandle = inst;

    DBG_LOG_INFO("Vulkan Instance Created");

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mHandle);
#endif

    QueyGPUs();
}

Instance::~Instance() {
    mHandle.destroy();
}

std::span<Type_STLString> Instance::GetExtensions() {
    return mEnabledExtensions;
}

PhysicalDevice& Instance::GetSuitableGPU(vk::SurfaceKHR surface) const {
    assert(!mGPUs.empty() && "No physical devices were found on the system.");

    // Find a discrete GPU
    for (auto& gpu : mGPUs) {
        if (gpu->GetProperties().deviceType
            == vk::PhysicalDeviceType::eDiscreteGpu) {
            // See if it work with the surface
            size_t queue_count = gpu->GetQueueFamilyProperties().size();
            for (uint32_t queue_idx = 0;
                 static_cast<size_t>(queue_idx) < queue_count; queue_idx++) {
                if (gpu->GetHandle().getSurfaceSupportKHR(queue_idx, surface)) {
                    return *gpu;
                }
            }
        }
    }

    // Otherwise just pick the first one
    DBG_LOG_INFO(
        "Couldn't find a discrete physical device, picking default GPU");
    return *mGPUs[0];
}

void Instance::QueyGPUs() {
    ::std::vector<vk::PhysicalDevice> gpus = mHandle.enumeratePhysicalDevices();
    if (gpus.empty()) {
        throw std::runtime_error(
            "Couldn't find a physical device that supports Vulkan.");
    }

    // Create gpus wrapper objects from the vk::PhysicalDevice's
    for (auto& physical_device : gpus) {
        mGPUs.push_back(MakeUnique<PhysicalDevice>(*this, physical_device));
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core