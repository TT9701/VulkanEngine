#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanInstance;
class SDLWindow;

class VulkanSurface {
public:
    VulkanSurface(VulkanInstance* instance, const SDLWindow* window);
    ~VulkanSurface();
    MOVABLE_ONLY(VulkanSurface);

public:
    VkSurfaceKHR const& GetHandle() const { return mSurface; }

private:
    VkSurfaceKHR CreateSurface(const SDLWindow* window) const;

private:
    VulkanInstance* pInstance;
    VkSurfaceKHR    mSurface;
};