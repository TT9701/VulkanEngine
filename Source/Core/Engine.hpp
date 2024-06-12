#pragma once

#define NOMINMAX
#include <vulkan/vulkan.hpp>

struct SDL_Window;

class VulkanEngine {
public:
    void Init();
    void Run();
    void Cleanup();

private:
    void Draw();

    void InitVulkan();
    void CreateInstance();

private:
    bool        mStopRendering {false};
    SDL_Window* mWindow {nullptr};
    int         mWindowWidth {1600};
    int         mWindowHeight {900};

    vk::Instance               mInstance {};
    vk::Device                 mDevice {};
    vk::PhysicalDevice         mPhysicalDevice {};
    vk::DebugUtilsMessengerEXT mDebugUtilsMessenger {};
};