#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanInstance;
class SDLWindow;

class VulkanSurface {
public:
    VulkanSurface(const SharedPtr<VulkanInstance>& instance,
                  const SharedPtr<SDLWindow>& window);
    ~VulkanSurface();
    MOVABLE_ONLY(VulkanSurface);

public:
    VkSurfaceKHR const& GetHandle() const { return mSurface; }

private:
    VkSurfaceKHR CreateSurface();

private:
    SharedPtr<VulkanInstance> pInstance;
    SharedPtr<SDLWindow> pWindow;

    VkSurfaceKHR mSurface;
};