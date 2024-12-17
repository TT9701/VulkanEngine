#pragma once
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/Logger.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

/**
 * @brief 
 */
struct DriverVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
};

/**
 * @brief 
 */
class PhysicalDevice {
public:
    PhysicalDevice(Instance& instance, vk::PhysicalDevice physicalDevice);

    ~PhysicalDevice() = default;
    CLASS_MOVABLE_ONLY(PhysicalDevice);

public:
    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDevice const* operator->() const;

    /**
     * @brief 
     * @return 
     */
    DriverVersion GetDriverVersion() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDeviceProperties const& GetProperties() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDeviceFeatures const& GetFeatures() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDeviceMemoryProperties const& GetMemoryProperties() const;

    /**
     * @brief 
     * @return 
     */
    ::std::span<vk::QueueFamilyProperties> GetQueueFamilyProperties();

    /**
     * @brief 
     * @param bits 
     * @param properties 
     * @param memoryTypeFound 
     * @return 
     */
    uint32_t GetMemoryType(uint32_t bits, vk::MemoryPropertyFlags properties,
                           vk::Bool32* memoryTypeFound = nullptr) const;

    /**
     * @brief 
     * @return 
     */
    void* GetExtensionFeatureChain() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDevice const& GetHandle() const;

    /**
     * @brief 
     * @param requestedExt 
     * @return 
     */
    bool IsExtensionSupported(const char* requestedExt) const;

    /**
     * @brief 
     * @param enable 
     */
    void SetHighPriorityGraphicsQueue(bool enable);

    /**
     * @brief 
     * @return 
     */
    bool HasHighPriorityGraphicsQueue() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDeviceFeatures const& GetRequestedFeatures() const;

    /**
     * @brief 
     * @return 
     */
    vk::PhysicalDeviceFeatures& GetMutableRequestedFeatures();

    /**
     * @brief 
     * @tparam StructureType 
     * @return 
     */
    template <class StructureType>
    StructureType GetProperties();

    /**
     * @brief 
     * @tparam StructureType 
     * @return 
     */
    template <class StructureType>
    StructureType GetExtensionFeatures();

    /**
     * @brief 
     * @tparam StructureType 
     * @return 
     */
    template <class StructureType>
    StructureType& AddExtensionFeatures();

    /**
     * @brief 
     * @tparam Feature 
     * @param flag 
     * @param featureName 
     * @param flagName 
     * @return 
     */
    template <typename Feature>
    vk::Bool32 RequestOptionalFeature(vk::Bool32 Feature::*flag,
                                      const char* featureName,
                                      const char* flagName);

    /**
     * @brief 
     * @tparam Feature 
     * @param flag 
     * @param featureName 
     * @param flagName 
     */
    template <typename Feature>
    void RequestRequiredFeature(vk::Bool32 Feature::*flag,
                                const char* featureName, const char* flagName);

private:
    Instance& mInstance;

    vk::PhysicalDevice mHandle;

    vk::PhysicalDeviceProperties2 mProperties;
    vk::PhysicalDeviceFeatures2 mFeatures;
    vk::PhysicalDeviceMemoryProperties2 mMemoryProperties;
    Type_STLVector<vk::QueueFamilyProperties> mQueueFamilyProperties;
    Type_STLVector<vk::ExtensionProperties> mDeviceExtensions;

    vk::PhysicalDeviceFeatures2 mRequestedFeatures;
    void* mLastRequestedExtensionFeature {nullptr};
    Type_STLMap<vk::StructureType, SharedPtr<void>> mExtensionFeatures;

    bool mHighPriorityGraphicsQueue {false};
};

template <class StructureType>
StructureType PhysicalDevice::GetProperties() {
    return mHandle
        .getProperties2<vk::PhysicalDeviceProperties2KHR, StructureType>()
        .template get<StructureType>();
}

template <class StructureType>
StructureType PhysicalDevice::GetExtensionFeatures() {
    return mHandle
        .getFeatures2KHR<vk::PhysicalDeviceFeatures2KHR, StructureType>()
        .template get<StructureType>();
}

template <class StructureType>
StructureType& PhysicalDevice::AddExtensionFeatures() {
    auto [it, success] = mExtensionFeatures.emplace(
        StructureType::structureType, MakeShared<StructureType>());
    if (success) {
        if (mLastRequestedExtensionFeature) {
            static_cast<StructureType*>(it->second.get())->pNext =
                mLastRequestedExtensionFeature;
        }
        mLastRequestedExtensionFeature = it->second.get();
    }

    return *static_cast<StructureType*>(it->second.get());
}

template <typename Feature>
vk::Bool32 PhysicalDevice::RequestOptionalFeature(vk::Bool32 Feature::*flag,
                                                  const char* featureName,
                                                  const char* flagName) {
    vk::Bool32 supported = GetExtensionFeatures<Feature>().*flag;
    if (supported) {
        AddExtensionFeatures<Feature>().*flag = true;
    } else {
        DBG_LOG_INFO("Requested optional feature %s::%s is not supported",
                     featureName, flagName);
    }
    return supported;
}

template <typename Feature>
void PhysicalDevice::RequestRequiredFeature(vk::Bool32 Feature::*flag,
                                            const char* featureName,
                                            const char* flagName) {
    if (GetExtensionFeatures<Feature>().*flag) {
        AddExtensionFeatures<Feature>().*flag = true;
    } else {
        throw std::runtime_error(std::string("Requested required feature ")
                                 + featureName + "::" + flagName
                                 + " is not supported");
    }
}

#define REQUEST_OPTIONAL_FEATURE(gpu, Feature, flag) \
    gpu.RequestOptionalFeature<Feature>(&Feature::flag, #Feature, #flag)
#define REQUEST_REQUIRED_FEATURE(gpu, Feature, flag) \
    gpu.RequestRequiredFeature<Feature>(&Feature::flag, #Feature, #flag)

}  // namespace IntelliDesign_NS::Vulkan::Core