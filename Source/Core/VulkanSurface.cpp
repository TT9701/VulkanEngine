#include "VulkanSurface.hpp"

#include <SDL2/SDL_vulkan.h>

#include "Utilities/Logger.hpp"
#include "VulkanInstance.hpp"
#include "Window.hpp"

VulkanSurface::VulkanSurface(const Type_SPInstance<VulkanInstance>& instance,
                             const Type_SPInstance<SDLWindow>& window)
    : pInstance(instance), pWindow(window), mSurface(CreateSurface()) {
    DBG_LOG_INFO("SDL Vulkan Surface Created");
}

VulkanSurface::~VulkanSurface() {
    pInstance->GetHandle().destroy(mSurface);
}

VkSurfaceKHR VulkanSurface::CreateSurface() {
    VkSurfaceKHR surface;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(pWindow->GetPtr(), pInstance->GetHandle(),
                             &surface);
#endif

    return surface;
}