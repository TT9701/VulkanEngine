#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanInstance;

class VulkanDebugUtils {
public:
    VulkanDebugUtils(VulkanInstance* instance);
    ~VulkanDebugUtils();
    MOVABLE_ONLY(VulkanDebugUtils);

public:
    vk::DebugUtilsMessengerEXT GetHandle() const { return mDebugMessenger; }

private:
    vk::DebugUtilsMessengerEXT CreateDebugMessenger();

private:
    VulkanInstance*            pInstance;
    vk::DebugUtilsMessengerEXT mDebugMessenger;
};