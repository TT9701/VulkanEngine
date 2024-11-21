#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Utilities/VulkanUtilities.h"

#include "VulkanResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;

std::vector<const char*> GetOptimalValidationLayers(
    ::std::span<vk::LayerProperties> supportedInstanceLayers);

class Instance : public VulkanResource<vk::Instance> {
public:
    Instance(::std::string_view& appName,
             ::std::unordered_map<const char*, bool> const& requiredExts,
             ::std::span<const char*> requiredValidationLayers,
             ::std::span<vk::LayerSettingEXT> requiredLayerSettings,
             uint32_t apiVersion);
    ~Instance() override;
    CLASS_NO_COPY_MOVE(Instance);

    ::std::span<const char*> GetExtensions();
    PhysicalDevice& GetFirstGPU();
    PhysicalDevice& GetSuitableGPU(vk::SurfaceKHR surface,
                                   bool headless_surface);

    bool IsEnabled(const char* extension) const;
    void QueryGPUs();

private:
    Type_STLVector<const char*> mEnabledExtensions;

#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT mDebugUtilsMessenger;
    vk::DebugReportCallbackEXT mDebugReportCallback;
#endif

    Type_STLVector<UniquePtr<PhysicalDevice>> mGPUs;
};

}  // namespace IntelliDesign_NS::Vulkan::Core