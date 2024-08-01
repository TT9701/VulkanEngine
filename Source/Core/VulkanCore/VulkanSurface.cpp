#include "VulkanSurface.hpp"

#include <SDL2/SDL_vulkan.h>

#include "Core/Utilities/Logger.hpp"
#include "VulkanInstance.hpp"
#include "Core/Platform/Window.hpp"

VulkanSurface::VulkanSurface(VulkanInstance* instance, const SDLWindow* window)
    : pInstance(instance), mSurface(CreateSurface(window)) {
    DBG_LOG_INFO("SDL Vulkan Surface Created");
}

VulkanSurface::~VulkanSurface() {
    pInstance->GetHandle().destroy(mSurface);
}

VkSurfaceKHR VulkanSurface::CreateSurface(const SDLWindow* window) const {
    VkSurfaceKHR surface;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(window->GetPtr(), pInstance->GetHandle(),
                             &surface);
#endif

    return surface;
}