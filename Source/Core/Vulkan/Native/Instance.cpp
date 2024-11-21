#include "Instance.h"

#include "Core/Utilities/VulkanUtilities.h"
#include "PhysicalDevice.h"

namespace {
bool LoadDispatcher = false;

using namespace IntelliDesign_NS::Core::MemoryPool;

// auto SetInstanceLayers(::std::span<Type_STLString> requestedLayers) {
//     if (!LoadDispatcher) {
// #if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
//         VULKAN_HPP_DEFAULT_DISPATCHER.init();
// #endif
//         LoadDispatcher = !LoadDispatcher;
//     }
//
//     auto instanceLayersProps = vk::enumerateInstanceLayerProperties();
//     Type_STLVector<Type_STLString> availableInstanceLayers {};
//     for (auto& prop : instanceLayersProps) {
//         availableInstanceLayers.push_back(prop.layerName.data());
//     }
//     return IntelliDesign_NS::Vulkan::Core::Utils::FilterStringList(
//         availableInstanceLayers, requestedLayers);
// }
//
// auto SetInstanceExtensions(std::span<Type_STLString> requestedExtensions) {
//     if (!LoadDispatcher) {
// #if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
//         VULKAN_HPP_DEFAULT_DISPATCHER.init();
// #endif
//         LoadDispatcher = !LoadDispatcher;
//     }
//
//     auto instanceExtensionProps = vk::enumerateInstanceExtensionProperties();
//     Type_STLVector<Type_STLString> availableInstanceExtensions {};
//     for (auto& prop : instanceExtensionProps) {
//         availableInstanceExtensions.push_back(prop.extensionName.data());
//     }
//     return IntelliDesign_NS::Vulkan::Core::Utils::FilterStringList(
//         availableInstanceExtensions, requestedExtensions);
// }

#ifndef NDEBUG
VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    // Log debug message
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        auto num = ::std::to_string(callback_data->messageIdNumber);
        DBG_LOG_INFO(num + " - " + callback_data->pMessageIdName + ": "
                     + callback_data->pMessage);
    } else if (message_severity
               & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        auto num = ::std::to_string(callback_data->messageIdNumber);
        DBG_LOG_INFO(num + " - " + callback_data->pMessageIdName + ": "
                     + callback_data->pMessage);
    }
    return VK_FALSE;
}
#endif

bool ValidateLayers(::std::span<const char*> required,
                    ::std::span<vk::LayerProperties> available) {
    for (auto layer : required) {
        bool found = false;
        for (auto& available_layer : available) {
            if (strcmp(available_layer.layerName, layer) == 0) {
                found = true;
                break;
            }
        }

        if (!found) {
            DBG_LOG_INFO(::std::string("Validation Layer not found: ") + layer);
            return false;
        }
    }

    return true;
}

std::vector<const char*> GetOptimalValidationLayers(
    std::span<vk::LayerProperties> supportedInstanceLayers) {
    std::vector<std::vector<const char*>> validation_layer_priority_list = {
        // The preferred validation layer is "VK_LAYER_KHRONOS_validation"
        {"VK_LAYER_KHRONOS_validation"},

        // Otherwise we fallback to using the LunarG meta layer
        {"VK_LAYER_LUNARG_standard_validation"},

        // Otherwise we attempt to enable the individual layers that compose the LunarG meta layer since it doesn't exist
        {
            "VK_LAYER_GOOGLE_threading",
            "VK_LAYER_LUNARG_parameter_validation",
            "VK_LAYER_LUNARG_object_tracker",
            "VK_LAYER_LUNARG_core_validation",
            "VK_LAYER_GOOGLE_unique_objects",
        },

        // Otherwise as a last resort we fallback to attempting to enable the LunarG core layer
        {"VK_LAYER_LUNARG_core_validation"}};

    for (auto& validation_layers : validation_layer_priority_list) {
        if (ValidateLayers(validation_layers, supportedInstanceLayers)) {
            return validation_layers;
        }

        DBG_LOG_INFO(
            "Couldn't enable validation layers (see log for error) - falling "
            "back");
    }

    // Else return nothing
    return {};
}

bool EnableExtension(const char* requiredExtName,
                     ::std::span<vk::ExtensionProperties> availableExts,
                     Type_STLVector<const char*>& enabledExtensions) {
    for (auto& availExtIt : availableExts) {
        if (strcmp(availExtIt.extensionName, requiredExtName) == 0) {
            auto it = std::find_if(
                enabledExtensions.begin(), enabledExtensions.end(),
                [requiredExtName](const char* enabled_ext_name) {
                    return strcmp(enabled_ext_name, requiredExtName) == 0;
                });
            if (it != enabledExtensions.end()) {
                // Extension is already enabled
            } else {
                DBG_LOG_INFO(::std::string {"Extension found, enabling it: "}
                             + requiredExtName);
                enabledExtensions.emplace_back(requiredExtName);
            }
            return true;
        }
    }

    DBG_LOG_INFO(::std::string {"Extension not found"} + requiredExtName);
    return false;
}

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

Instance::Instance(std::string_view& appName,
                   std::unordered_map<const char*, bool> const& requiredExts,
                   std::span<const char*> requiredValidationLayers,
                   std::span<vk::LayerSettingEXT> requiredLayerSettings,
                   uint32_t apiVersion) {
    if (!LoadDispatcher) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif
        LoadDispatcher = !LoadDispatcher;
    }

    std::vector<vk::ExtensionProperties> availableExts =
        vk::enumerateInstanceExtensionProperties();

#ifndef NDEBUG
    const bool hasDebugUtils = EnableExtension(
        vk::EXTDebugUtilsExtensionName, availableExts, mEnabledExtensions);
    if (!hasDebugUtils) {
        DBG_LOG_INFO(::std::string {vk::EXTDebugUtilsExtensionName}
                     + "is not available; disable debug reporting.");
    }
#endif

    // If using VK_EXT_headless_surface, we still create and use a surface
    mEnabledExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

    // VK_KHR_get_physical_device_properties2 is a prerequisite of VK_KHR_performance_query
    // which will be used for stats gathering where available.
    EnableExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                    availableExts, mEnabledExtensions);

    auto extensionError = false;
    for (auto extension : requiredExts) {
        auto extName = extension.first;
        auto extIsOptional = extension.second;
        if (!EnableExtension(extName, availableExts, mEnabledExtensions)) {
            if (extIsOptional) {
                DBG_LOG_INFO(
                    ::std::string {"Optional instance extension not available, "
                                   "some features may be disabled: "}
                    + extName);
            } else {
                DBG_LOG_INFO(::std::string {"Required instance extension not "
                                            "available, cannot run: "}
                             + extName);
                extensionError = true;
            }
            extensionError = extensionError || !extIsOptional;
        }
    }

    if (extensionError) {
        throw std::runtime_error("Required instance extensions are missing.");
    }

    std::vector<vk::LayerProperties> supportedValidationLayers =
        vk::enumerateInstanceLayerProperties();

    std::vector<const char*> requestedValidationLayers(
        requiredValidationLayers.size());
    for (uint32_t i = 0; i < requestedValidationLayers.size(); ++i) {
        requestedValidationLayers[i] = requiredValidationLayers[i];
    }

#ifndef NDEBUG
    std::vector<const char*> optimalValidationLayers =
        GetOptimalValidationLayers(supportedValidationLayers);
    requestedValidationLayers.insert(requestedValidationLayers.end(),
                                     optimalValidationLayers.begin(),
                                     optimalValidationLayers.end());
#endif

    if (ValidateLayers(requestedValidationLayers, supportedValidationLayers)) {
        DBG_LOG_INFO("Enabled Validation Layers:");
        for (const auto& layer : requestedValidationLayers) {
            DBG_LOG_INFO(::std::string {"	\t"} + layer);
        }
    } else {
        throw std::runtime_error("Required validation layers are missing.");
    }

    vk::ApplicationInfo appInfo {appName.data(), 0, "Vulkan", 0, apiVersion};

    vk::InstanceCreateInfo instanceInfo {
        {}, &appInfo, requestedValidationLayers, mEnabledExtensions};

#ifndef NDEBUG
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
    messengerInfo
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
        .setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
#if defined(VK_EXT_device_address_binding_report)
            vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
#endif
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(&DebugUtilsMessengerCallback);

    instanceInfo.setPNext(&messengerInfo);
#endif

    vk::LayerSettingsCreateInfoEXT layerSettingsCreateInfo;

    // If layer settings extension enabled by sample, then activate layer settings during instance creation
    if (std::ranges::find(mEnabledExtensions,
                          VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)
        != mEnabledExtensions.end()) {
        layerSettingsCreateInfo.settingCount =
            static_cast<uint32_t>(requiredLayerSettings.size());
        layerSettingsCreateInfo.pSettings = requiredLayerSettings.data();
        layerSettingsCreateInfo.pNext = instanceInfo.pNext;
        instanceInfo.pNext = &layerSettingsCreateInfo;
    }

    // Create the Vulkan instance
    auto& handle = GetHandle();
    handle = vk::createInstance(instanceInfo);

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(handle);
#endif

#ifndef NDEBUG
    mDebugUtilsMessenger = handle.createDebugUtilsMessengerEXT(messengerInfo);
#endif

    QueryGPUs();
}

// Instance::Instance(std::span<Type_STLString> requestedInstanceLayers,
//                    std::span<Type_STLString> requestedInstanceExtensions)
//     : mEnabledInstanceLayers(SetInstanceLayers(requestedInstanceLayers)),
//       mEnabledInstanceExtensions(
//           SetInstanceExtensions(requestedInstanceExtensions)),
//       mInstance(CreateInstance()) {
//     DBG_LOG_INFO("Vulkan Instance Created");
//
// #if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
//     VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance);
// #endif
// }

Instance::~Instance() {
    GetHandle().destroy();
}

std::span<const char*> Instance::GetExtensions() {
    return mEnabledExtensions;
}

PhysicalDevice& Instance::GetFirstGPU() {
    assert(!mGPUs.empty() && "No physical devices were found on the system.");

    // Find a discrete GPU
    for (auto& gpu : mGPUs) {
        if (gpu->GetProperties().deviceType
            == vk::PhysicalDeviceType::eDiscreteGpu) {
            return *gpu;
        }
    }

    // Otherwise just pick the first one
    DBG_LOG_INFO(
        "Couldn't find a discrete physical device, picking default GPU");
    return *mGPUs[0];
}

PhysicalDevice& Instance::GetSuitableGPU(vk::SurfaceKHR surface,
                                         bool headless_surface) {
    assert(!mGPUs.empty() && "No physical devices were found on the system.");

    if (headless_surface) {
        DBG_LOG_INFO(
            "Using headless surface with multiple GPUs. Considered explicitly "
            "selecting the target GPU.");
    }

    // Find a discrete GPU
    for (auto& gpu : mGPUs) {
        if (gpu->GetProperties().deviceType
            == vk::PhysicalDeviceType::eDiscreteGpu) {
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

bool Instance::IsEnabled(const char* extension) const {
    return std::ranges::find_if(mEnabledExtensions,
                                [extension](const char* enabled_extension) {
                                    return strcmp(extension, enabled_extension)
                                        == 0;
                                })
        != mEnabledExtensions.end();
}

void Instance::QueryGPUs() {
    auto handle = GetHandle();
    std::vector<vk::PhysicalDevice> physical_devices =
        handle.enumeratePhysicalDevices();
    if (physical_devices.empty()) {
        throw std::runtime_error(
            "Couldn't find a physical device that supports Vulkan.");
    }

    // Create gpus wrapper objects from the vk::PhysicalDevice's
    for (auto& physical_device : physical_devices) {
        mGPUs.push_back(MakeUnique<PhysicalDevice>(*this, physical_device));
    }
}

// vk::Instance Instance::CreateInstance() {
//     if (!LoadDispatcher) {
// #if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
//         VULKAN_HPP_DEFAULT_DISPATCHER.init();
// #endif
//         LoadDispatcher = !LoadDispatcher;
//     }
//
//     Type_STLVector<const char*> enabledLayersCStr(
//         mEnabledInstanceLayers.size());
//     for (int i = 0; i < mEnabledInstanceLayers.size(); ++i) {
//         enabledLayersCStr[i] = mEnabledInstanceLayers[i].c_str();
//     }
//
//     Type_STLVector<const char*> enabledExtensionsCStr(
//         mEnabledInstanceExtensions.size());
//     for (int i = 0; i < mEnabledInstanceExtensions.size(); ++i) {
//         enabledExtensionsCStr[i] = mEnabledInstanceExtensions[i].c_str();
//     }
//
//     vk::ApplicationInfo appInfo {};
//     appInfo.setPEngineName("Vulkan Engine")
//         .setPApplicationName("Fun")
//         .setEngineVersion(1u)
//         .setApplicationVersion(1u)
//         .setApiVersion(VK_API_VERSION_1_3);
//
//     vk::InstanceCreateInfo instanceCreateInfo {};
//     instanceCreateInfo.setPApplicationInfo(&appInfo)
//         .setPEnabledLayerNames(enabledLayersCStr)
//         .setPEnabledExtensionNames(enabledExtensionsCStr);
//
//     auto inst = vk::createInstance(instanceCreateInfo);
//
//     return inst;
// }

}  // namespace IntelliDesign_NS::Vulkan::Core