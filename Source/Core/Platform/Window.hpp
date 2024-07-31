#pragma once

#include <SDL2/SDL_events.h>
#include "Core/Utilities/Defines.hpp"

class SDLWindow {
public:
    SDLWindow(int width = 1600, int height = 900);
    ~SDLWindow();
    MOVABLE_ONLY(SDLWindow);

public:
    void PollEvents(bool& quit, bool& stopRendering);

public:
    std::vector<std::string> GetVulkanInstanceExtension() const;

    SDL_Window* GetPtr() const { return mWindow; }

    int GetWidth() const { return mWidth; }

    int GetHeight() const { return mHeight; }

private:
    int mWidth;
    int mHeight;

    SDL_Window* mWindow;
    SDL_Event*  mEvent {nullptr};
};