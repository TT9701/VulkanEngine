#pragma once

#include <functional>
#include <string>
#include <vector>

#include <SDL2/SDL_events.h>
#include <SDL_timer.h>
#include "Core/Utilities/Defines.h"

class SDLWindow {
public:
    SDLWindow(const char* name, int width, int height);
    ~SDLWindow();
    MOVABLE_ONLY(SDLWindow);

public:
    void PollEvents(bool& quit, bool& stopRendering,
                    ::std::function<void(SDL_Event*)>&& eventFunc,
                    ::std::function<void()>&& onWindowResized);

public:
    std::vector<std::string> GetVulkanInstanceExtension() const;

    SDL_Window* GetPtr() const { return mWindow; }

    int GetWidth() const { return mWidth; }

    int GetHeight() const { return mHeight; }

    SDL_Event* GetEvent() const { return mEvent; }

private:
    ::std::string mName;
    int mWidth;
    int mHeight;

    SDL_Window* mWindow;
    SDL_Event* mEvent {nullptr};
};