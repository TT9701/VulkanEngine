#include "Surface.h"

#include <SDL2/SDL_vulkan.h>

#include "Core/Platform/Window.h"
#include "Core/Utilities/Logger.h"
#include "Instance.h"

namespace IntelliDesign_NS::Vulkan::Core {

Surface::Surface(Instance& instance, const SDLWindow& window)
    : mInstance(instance) {
    VkSurfaceKHR surface;

#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(window.GetPtr(), instance.GetHandle(), &surface);
#endif

    mHandle = surface;

    DBG_LOG_INFO("SDL Vulkan Surface Created");
}

Surface::~Surface() {
    mInstance.GetHandle().destroy(mHandle);
}

}  // namespace IntelliDesign_NS::Vulkan::Core