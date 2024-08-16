#include "Instance.hpp"

#include "Core/Utilities/Logger.hpp"

namespace {
bool LoadDispatcher = false;

auto SetInstanceLayers(::std::span<::std::string> requestedLayers) {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    auto instanceLayersProps = vk::enumerateInstanceLayerProperties();
    ::std::vector<::std::string> availableInstanceLayers {};
    for (auto& prop : instanceLayersProps) {
        availableInstanceLayers.push_back(prop.layerName);
    }
    return Utils::FilterStringList(availableInstanceLayers, requestedLayers);
}

auto SetInstanceExtensions(std::span<std::string> requestedExtensions) {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    auto instanceExtensionProps = vk::enumerateInstanceExtensionProperties();
    ::std::vector<::std::string> availableInstanceExtensions {};
    for (auto& prop : instanceExtensionProps) {
        availableInstanceExtensions.push_back(prop.extensionName);
    }
    return Utils::FilterStringList(availableInstanceExtensions,
                                   requestedExtensions);
}

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

Instance::Instance(
    std::span<std::string> requestedInstanceLayers,
    std::span<std::string> requestedInstanceExtensions)
    : mEnabledInstanceLayers(SetInstanceLayers(requestedInstanceLayers)),
      mEnabledInstanceExtensions(
          SetInstanceExtensions(requestedInstanceExtensions)),
      mInstance(CreateInstance()) {
    DBG_LOG_INFO("Vulkan Instance Created");

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance);
#endif
}

Instance::~Instance() {
    mInstance.destroy();
}

vk::Instance Instance::CreateInstance() {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    ::std::vector<const char*> enabledLayersCStr(mEnabledInstanceLayers.size());
    for (int i = 0; i < mEnabledInstanceLayers.size(); ++i) {
        enabledLayersCStr[i] = mEnabledInstanceLayers[i].c_str();
    }

    ::std::vector<const char*> enabledExtensionsCStr(
        mEnabledInstanceExtensions.size());
    for (int i = 0; i < mEnabledInstanceExtensions.size(); ++i) {
        enabledExtensionsCStr[i] = mEnabledInstanceExtensions[i].c_str();
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

    return inst;
}

}  // namespace IntelliDesign_NS::Vulkan::Core