#pragma once

#include <SDL2/SDL_events.h>
#include "Core/Utilities/Defines.hpp"

class SDLWindow {
public:
    SDLWindow(int width = 1600, int height = 900);
    ~SDLWindow();
    MOVABLE_ONLY(SDLWindow);

public:
    void PollEvents(bool& quit, bool& stopRendering, ::std::function<void(SDL_Event*)>&& func);

public:
    std::vector<std::string> GetVulkanInstanceExtension() const;

    SDL_Window* GetPtr() const { return mWindow; }

    int GetWidth() const { return mWidth; }

    int GetHeight() const { return mHeight; }

    SDL_Event* GetEvent() const { return mEvent; }

private:
    int mWidth;
    int mHeight;

    SDL_Window* mWindow;
    SDL_Event*  mEvent {nullptr};
};