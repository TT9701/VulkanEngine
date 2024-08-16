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
    vk::SurfaceKHR const& GetHandle() const { return mSurface; }

private:
    vk::SurfaceKHR CreateSurface(const SDLWindow* window) const;

private:
    VulkanInstance* pInstance;
    vk::SurfaceKHR    mSurface;
};