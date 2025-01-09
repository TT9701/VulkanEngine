#pragma once

#include <vulkan/vulkan.hpp>
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

#include <array>
#include <functional>
#include <vector>

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Swapchain;

class GUI {
public:
    GUI(VulkanContext& context, Swapchain& swapchain, SDLWindow& window);
    ~GUI();

public:
    void PollEvent(const SDL_Event* event);
    void BeginFrame();
    void Draw(vk::CommandBuffer cmd);

    GUI& AddContext(::std::function<void()>&& ctx);

private:
    void PrepareContext();
    void CreateDescPool();

private:
    VulkanContext& mContext;
    Swapchain& mSwapchain;
    SDLWindow& mWindow;

    ::std::array<vk::DescriptorPoolSize, 11> mPoolSizes;
    vk::DescriptorPool mDescPool;

    ::std::vector<::std::function<void()>> mUIContexts;
};

}  // namespace IntelliDesign_NS::Vulkan::Core