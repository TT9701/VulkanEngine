#include "Surface.h"

#include <SDL2/SDL_vulkan.h>

#include "Core/Utilities/Logger.h"
#include "Instance.h"
#include "Core/Platform/Window.h"

namespace IntelliDesign_NS::Vulkan::Core {

Surface::Surface(Instance* instance, const SDLWindow* window)
    : pInstance(instance), mSurface(CreateSurface(window)) {
    DBG_LOG_INFO("SDL Vulkan Surface Created");
}

Surface::~Surface() {
    pInstance->GetHandle().destroy(mSurface);
}

vk::SurfaceKHR Surface::CreateSurface(const SDLWindow* window) const {
    VkSurfaceKHR surface;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(window->GetPtr(), pInstance->GetHandle(),
                             &surface);
#endif

    return surface;
}

}  // namespace IntelliDesign_NS::Vulkan::Core