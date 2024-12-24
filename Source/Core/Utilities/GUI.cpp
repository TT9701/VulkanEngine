#include "GUI.h"

#include "Core/Platform/Window.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Core/Vulkan/Native/Swapchain.h"

namespace IntelliDesign_NS::Vulkan::Core {

GUI::GUI(VulkanContext& context, Swapchain& swapchain, SDLWindow& window)
    : mContext(context), mSwapchain(swapchain), mWindow(window) {
    CreateDescPool();
    PrepareContext();
}

GUI::~GUI() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    mContext.GetDevice()->destroy(mDescPool);
}

void GUI::PollEvent(const SDL_Event* event) {
    ImGui_ImplSDL2_ProcessEvent(event);
}

void GUI::BeginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // ImGui::ShowDemoWindow();

    for (auto const& ctx : mUIContexts)
        ctx();

    ImGui::Render();
}

void GUI::Draw(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo attachmentInfo {};
    attachmentInfo.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setImageView(mSwapchain.GetCurrentImage().GetTexViewHandle());

    vk::RenderingInfo renderingInfo {};
    renderingInfo.setColorAttachments(attachmentInfo)
        .setLayerCount(1)
        .setRenderArea(vk::Rect2D {{0, 0}, mSwapchain.GetExtent2D()});

    cmd.beginRendering(renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    cmd.endRendering();
}

void GUI::AddContext(std::function<void()>&& ctx) {
    mUIContexts.push_back(::std::move(ctx));
}

void GUI::PrepareContext() {
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForVulkan(mWindow.GetPtr());

    auto swapchainImageFormat = mSwapchain.GetFormat();

    ImGui_ImplVulkan_InitInfo info {};
    info.Instance = mContext.GetInstance().GetHandle();
    info.PhysicalDevice = mContext.GetPhysicalDevice().GetHandle();
    info.Device = mContext.GetDevice().GetHandle();
    info.Queue = mContext.GetQueue(QueueType::Graphics).GetHandle();
    info.DescriptorPool = mDescPool;
    info.MinImageCount = 2;
    info.ImageCount = mSwapchain.GetImageCount();
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats =
        reinterpret_cast<VkFormat*>(&swapchainImageFormat);
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&info);

    float sizePixels = 16.0;
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("../../Resources/fonts/Roboto-Medium.ttf",
                                 sizePixels);
}

void GUI::CreateDescPool() {
    mPoolSizes = ::std::array<vk::DescriptorPoolSize, 11> {
        vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 100},
        {vk::DescriptorType::eCombinedImageSampler, 100},
        {vk::DescriptorType::eSampledImage, 100},
        {vk::DescriptorType::eStorageImage, 100},
        {vk::DescriptorType::eUniformTexelBuffer, 100},
        {vk::DescriptorType::eStorageTexelBuffer, 100},
        {vk::DescriptorType::eUniformBuffer, 100},
        {vk::DescriptorType::eStorageBuffer, 100},
        {vk::DescriptorType::eUniformBufferDynamic, 100},
        {vk::DescriptorType::eStorageBufferDynamic, 100},
        {vk::DescriptorType::eInputAttachment, 100}};

    vk::DescriptorPoolCreateInfo dpInfo {};
    dpInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(100)
        .setPoolSizes(mPoolSizes);

    mDescPool = mContext.GetDevice()->createDescriptorPool(dpInfo);
}

}  // namespace IntelliDesign_NS::Vulkan::Core