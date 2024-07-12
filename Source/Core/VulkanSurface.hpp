#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanHelper.hpp"

class VulkanInstance;
class SDLWindow;

class VulkanSurface {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanSurface(const Type_SPInstance<VulkanInstance>& instance,
                  const Type_SPInstance<SDLWindow>& window);
    ~VulkanSurface();

    VkSurfaceKHR const& GetHandle() const { return mSurface; }

private:
    VkSurfaceKHR CreateSurface();

private:
    Type_SPInstance<VulkanInstance> pInstance;
    Type_SPInstance<SDLWindow> pWindow;

    VkSurfaceKHR mSurface;
};