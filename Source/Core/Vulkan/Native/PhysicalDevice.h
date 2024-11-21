#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "VulkanResource.h"

#include <vulkan/vulkan.hpp>

#include "Instance.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

struct DriverVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
};

class PhysicalDevice : public VulkanResource<vk::PhysicalDevice> {
public:
    // PhysicalDevice(Instance* instance,
    //                      vk::QueueFlags requestedQueueTypes);

    PhysicalDevice(Instance& instance, vk::PhysicalDevice gpu);

    ~PhysicalDevice() = default;
    CLASS_NO_COPY_MOVE(PhysicalDevice);

public:
    DriverVersion GetDriverVersion() const;

    void* GetExtensionFeatureChain() const;

    bool IsExtensionSupported(::std::string const& requested_extension) const;

    const vk::PhysicalDeviceFeatures& GetFeatures() const;

    Instance& GetInstance() const;

    const vk::PhysicalDeviceMemoryProperties& GetMemoryProperties() const;

    uint32_t GetMemoryType(uint32_t bits, vk::MemoryPropertyFlags properties,
                           vk::Bool32* memory_type_found = nullptr) const;

    const vk::PhysicalDeviceProperties& GetProperties() const;

    const Type_STLVector<vk::QueueFamilyProperties>& GetQueueFamilyProperties()
        const;

    const vk::PhysicalDeviceFeatures GetRequestedFeatures() const;

    vk::PhysicalDeviceFeatures& GetMutableRequestedFeatures();

    template <typename StructureType>
    StructureType GetExtensionFeatures() {
        if (!mInstance.IsEnabled(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            throw std::runtime_error(
                "Couldn't request feature from device as "
                + std::string(
                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
                + " isn't enabled!");
        }

        // Get the extension feature
        return GetHandle()
            .getFeatures2KHR<vk::PhysicalDeviceFeatures2KHR, StructureType>()
            .template get<StructureType>();
    }

    template <typename StructureType>
    StructureType& AddExtensionFeatures() {
        if (!mInstance.IsEnabled(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            throw std::runtime_error(
                "Couldn't request feature from device as "
                + std::string(
                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
                + " isn't enabled!");
        }

        // Add an (empty) extension features into the map of extension features
        auto [it, added] = mExtensionFeatures.insert(
            {StructureType::structureType, std::make_shared<StructureType>()});
        if (added) {
            // if it was actually added, also add it to the structure chain
            if (mLastRequestedExtensionFeature) {
                static_cast<StructureType*>(it->second.get())->pNext =
                    mLastRequestedExtensionFeature;
            }
            mLastRequestedExtensionFeature = it->second.get();
        }

        return *static_cast<StructureType*>(it->second.get());
    }

    template <typename Feature>
    vk::Bool32 RequestOptionalFeature(vk::Bool32 Feature::*flag,
                                      std::string const& featureName,
                                      std::string const& flagName) {
        vk::Bool32 supported = GetExtensionFeatures<Feature>().*flag;
        if (supported) {
            AddExtensionFeatures<Feature>().*flag = true;
        } else {
            DBG_LOG_INFO(
                ::std::string {"Requested optional feature is not supported"}
                + featureName + flagName);
        }
        return supported;
    }

    template <typename Feature>
    void RequestRequiredFeature(vk::Bool32 Feature::*flag,
                                std::string const& featureName,
                                std::string const& flagName) {
        if (GetExtensionFeatures<Feature>().*flag) {
            AddExtensionFeatures<Feature>().*flag = true;
        } else {
            throw std::runtime_error(std::string("Requested required feature <")
                                     + featureName + "::" + flagName
                                     + "> is not supported");
        }
    }

    void SetHighPriorityGraphicsQueueEnable(bool enable);

    bool HasHighPriorityGraphicsQueue() const;

private:
    Instance& mInstance;

    vk::PhysicalDeviceFeatures mFeatures;
    Type_STLVector<vk::ExtensionProperties> mDeviceExts;

    vk::PhysicalDeviceProperties mProperties;
    vk::PhysicalDeviceMemoryProperties mMemoryProperties;
    Type_STLVector<vk::QueueFamilyProperties> mQueueFamilyProperties;

    vk::PhysicalDeviceFeatures mRequestedFeatures;

    void* mLastRequestedExtensionFeature {nullptr};

    std::map<vk::StructureType, std::shared_ptr<void>> mExtensionFeatures;

    bool mHighPriorityGraphicsQueue {false};
};

#define REQUEST_OPTIONAL_FEATURE(gpu, Feature, flag) \
    gpu.RequestOptionalFeature<Feature>(&Feature::flag, #Feature, #flag)
#define REQUEST_REQUIRED_FEATURE(gpu, Feature, flag) \
    gpu.RequestRequiredFeature<Feature>(&Feature::flag, #Feature, #flag)

}  // namespace IntelliDesign_NS::Vulkan::Core