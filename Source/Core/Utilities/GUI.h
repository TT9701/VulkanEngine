#pragma once

#include <array>
#include <functional>
#include <vector>

#include <ImGuiFileDialog.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <vulkan/vulkan.hpp>

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Swapchain;
class RenderFrame;

class GUI {
public:
    GUI(VulkanContext& context, Swapchain& swapchain, SDLWindow& window);
    ~GUI();

public:
    void PollEvent(const SDL_Event* event);
    void BeginFrame(RenderFrame& frame);
    void Draw(vk::CommandBuffer cmd);

    GUI& AddContext(::std::function<void()>&& ctx);

    GUI& AddFrameRelatedContext(::std::function<void(RenderFrame&)>&& ctx);

    bool WantCaptureKeyboard() const;
    bool WantCaptureMouse() const;

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
    ::std::vector<::std::function<void(RenderFrame&)>> mFrameContexts;
};

}  // namespace IntelliDesign_NS::Vulkan::Core