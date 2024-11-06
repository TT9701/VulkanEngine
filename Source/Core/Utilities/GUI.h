#pragma once

#include <vulkan/vulkan.hpp>
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

#include <array>
#include <vector>
#include <functional>

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Swapchain;

class GUI {
public:
    GUI(Context* context, Swapchain* swapchain, SDLWindow* window);
    ~GUI();

public:
    void PollEvent(const SDL_Event* event);
    void BeginFrame();
    void Draw(vk::CommandBuffer cmd);

    void AddContext(::std::function<void()>&& ctx);

private:
    void PrepareContext();
    void CreateDescPool();

private:
    Context* pContext;
    Swapchain* pSwapchain;
    SDLWindow* pWindow;

    ::std::array<vk::DescriptorPoolSize, 11> mPoolSizes;
    vk::DescriptorPool mDescPool;

    ::std::vector<::std::function<void()>> mUIContexts;
};

}  // namespace IntelliDesign_NS::Vulkan::Core