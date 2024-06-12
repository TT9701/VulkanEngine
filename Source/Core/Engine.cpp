#include "Engine.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <thread>

#include "VulkanHelper.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

void VulkanEngine::Init() {
    SDL_Init(SDL_INIT_VIDEO);
    
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
    
    mWindow = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, mWindowWidth,
    mWindowHeight, window_flags);

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif

    InitVulkan();
}

void VulkanEngine::Run() {
    SDL_Event sdlEvent;
    
    bool bQuit = false;
    
    while (!bQuit) {
        while (SDL_PollEvent(&sdlEvent) != 0) {
            if (sdlEvent.type == SDL_QUIT)
                bQuit = true;
            if (sdlEvent.type == SDL_WINDOWEVENT) {
                if (sdlEvent.window.event == SDL_WINDOWEVENT_MINIMIZED)
                    mStopRendering = true;
                if (sdlEvent.window.event == SDL_WINDOWEVENT_RESTORED)
                    mStopRendering = false;
            }
        }
        if (mStopRendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            Draw();
        }
    }
}

void VulkanEngine::Cleanup() {}

void VulkanEngine::Draw() {}

void VulkanEngine::InitVulkan() {
    CreateInstance();
}

void VulkanEngine::CreateInstance() {
    vk::ApplicationInfo    appInfo {"Vulkan Engine", 1u, "Fun", 1u, VK_API_VERSION_1_3};
    vk::InstanceCreateInfo instanceCreateInfo {{}, &appInfo};
    VK_CHECK(vk::createInstance(&instanceCreateInfo, nullptr, &mInstance));
}
