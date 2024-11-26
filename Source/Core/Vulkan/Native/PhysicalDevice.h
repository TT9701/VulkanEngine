#pragma once
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

class PhysicalDevice {
public:
    PhysicalDevice(Instance* instance,
                         vk::QueueFlags requestedQueueTypes);
    ~PhysicalDevice() = default;
    CLASS_MOVABLE_ONLY(PhysicalDevice);

public:
    vk::PhysicalDevice const& GetHandle() const { return mPhysicalDevice; }

    std::optional<uint32_t> GetGraphicsQueueFamilyIndex() const {
        return mGraphicsFamilyIndex;
    }

    std::optional<uint32_t> GetComputeQueueFamilyIndex() const {
        return mComputeFamilyIndex;
    }

    std::optional<uint32_t> GetTransferQueueFamilyIndex() const {
        return mTransferFamilyIndex;
    }

    uint32_t GetGraphicsQueueCount() const { return mGraphicsQueueCount; }

    uint32_t GetComputeQueueCount() const { return mComputeQueueCount; }

    uint32_t GetTransferQueueCount() const { return mTransferQueueCount; }

private:
    vk::PhysicalDevice PickPhysicalDevice(vk::QueueFlags requestedQueueTypes);

    void SetQueueFamlies(vk::QueueFlags requestedQueueTypes);

private:
    Instance* pInstance;

    vk::PhysicalDevice mPhysicalDevice;

    std::optional<uint32_t> mGraphicsFamilyIndex;
    uint32_t mGraphicsQueueCount = 0;
    std::optional<uint32_t> mComputeFamilyIndex;
    uint32_t mComputeQueueCount = 0;
    std::optional<uint32_t> mTransferFamilyIndex;
    uint32_t mTransferQueueCount = 0;
};

}  // namespace IntelliDesign_NS::Vulkan::Core