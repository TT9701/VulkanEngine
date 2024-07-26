#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanInstance;

class VulkanDebugUtils {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanDebugUtils(Type_SPInstance<VulkanInstance> const& instance);
    ~VulkanDebugUtils();
    MOVABLE_ONLY(VulkanDebugUtils);

public:
    vk::DebugUtilsMessengerEXT const& GetHandle() const {
        return mDebugMessenger;
    }

private:
    vk::DebugUtilsMessengerEXT CreateDebugMessenger();

private:
    Type_SPInstance<VulkanInstance> pInstance;
    vk::DebugUtilsMessengerEXT mDebugMessenger;
};