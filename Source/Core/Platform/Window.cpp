#include "Window.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include "Core/Utilities/Logger.h"

namespace {

SDL_Window* CreateWindow(const char* name, int width, int height) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    return SDL_CreateWindow(name, SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED, width, height,
                            window_flags);
}

}  // namespace

SDLWindow::SDLWindow(const char* name, int width, int height)
    : mName(name),
      mWidth(width),
      mHeight(height),
      mWindow(CreateWindow(name, mWidth, mHeight)),
      mEvent(new SDL_Event()) {

    DBG_LOG_INFO("SDL_Window Created. Width: %d, Height: %d.", mWidth, mHeight);
}

SDLWindow::~SDLWindow() {
    delete mEvent;
    SDL_DestroyWindow(mWindow);
}

void SDLWindow::PollEvents(bool& quit, bool& stopRendering,
                           ::std::function<void(SDL_Event*)>&& eventFunc,
                           ::std::function<void()>&& onWindowResized) {
    auto func = ::std::move(eventFunc);
    auto onResized = ::std::move(onWindowResized);

    while (SDL_PollEvent(mEvent)) {
        switch (mEvent->type) {
            case SDL_QUIT: quit = true; break;
            case SDL_WINDOWEVENT:
                switch (mEvent->window.event) {
                    case SDL_WINDOWEVENT_MINIMIZED: stopRendering = true; break;
                    case SDL_WINDOWEVENT_RESTORED: stopRendering = false; break;
                    case SDL_WINDOWEVENT_SIZE_CHANGED:
                        SDL_GetWindowSize(mWindow, &mWidth, &mHeight);
                        onResized();
                        break;
                    default: break;
                }
                break;
            default: break;
        }
        func(mEvent);
    }
}

std::vector<IntelliDesign_NS::Core::MemoryPool::Type_STLString> SDLWindow::GetVulkanInstanceExtension() const {
    uint32_t count {0};
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count, nullptr);
    ::std::vector<const char*> requestedExtensions(count);
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count,
                                     requestedExtensions.data());

    std::vector<IntelliDesign_NS::Core::MemoryPool::Type_STLString> result {};
    result.reserve(requestedExtensions.size());
    for (auto& ext : requestedExtensions) {
        result.emplace_back(ext);
    }
    return result;
}