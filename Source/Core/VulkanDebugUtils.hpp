#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanHelper.hpp"

class VulkanInstance;

class VulkanDebugUtils {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanDebugUtils(Type_SPInstance<VulkanInstance> const& instance);
    ~VulkanDebugUtils();

public:
    vk::DebugUtilsMessengerEXT const& GetDebugMessenger() const {
        return mDebugMessenger;
    }

private:
    vk::DebugUtilsMessengerEXT CreateDebugMessenger();

private:
    Type_SPInstance<VulkanInstance> pInstance;
    vk::DebugUtilsMessengerEXT mDebugMessenger;
};