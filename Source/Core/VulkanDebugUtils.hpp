#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanInstance;

class VulkanDebugUtils {
public:
    VulkanDebugUtils(SharedPtr<VulkanInstance> const& instance);
    ~VulkanDebugUtils();
    MOVABLE_ONLY(VulkanDebugUtils);

public:
    vk::DebugUtilsMessengerEXT const& GetHandle() const {
        return mDebugMessenger;
    }

private:
    vk::DebugUtilsMessengerEXT CreateDebugMessenger();

private:
    SharedPtr<VulkanInstance> pInstance;
    vk::DebugUtilsMessengerEXT mDebugMessenger;
};