#pragma once
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanInstance;

class VulkanPhysicalDevice {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanPhysicalDevice(Type_SPInstance<VulkanInstance> const& instance,
                         vk::QueueFlags requestedQueueTypes);
    ~VulkanPhysicalDevice() = default;
    MOVABLE_ONLY(VulkanPhysicalDevice);

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
    Type_SPInstance<VulkanInstance> pInstance;

    vk::PhysicalDevice mPhysicalDevice;

    std::optional<uint32_t> mGraphicsFamilyIndex;
    uint32_t mGraphicsQueueCount = 0;
    std::optional<uint32_t> mComputeFamilyIndex;
    uint32_t mComputeQueueCount = 0;
    std::optional<uint32_t> mTransferFamilyIndex;
    uint32_t mTransferQueueCount = 0;
};