#include "Application.h"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace IntelliDesign_NS::Vulkan::Core {

Application::Application(ApplicationSpecification const& spec)
    : mWindow(CreateSDLWindow(spec.Name.c_str(), spec.width, spec.height)),
      mVulkanContext(CreateContext()),
      mSwapchain(CreateSwapchain()),
      mRenderResMgr(CreateRenderResourceManager()),
      mCmdMgr(CreateCommandManager()),
      mPipelineMgr(CreatePipelineManager()),
      mShaderMgr(CreateShaderManager()),
      mDGCSequenceMgr(CreateDGCSeqManager()),
      mGui(GetVulkanContext(), GetSwapchain(), GetSDLWindow())
#ifdef CUDA_VULKAN_INTEROP
      ,
      mCUDAExternalImage(CreateExternalImage())
#endif
{
#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif
}

Application::~Application() {
    mVulkanContext->GetDevice()->waitIdle();
}

void Application::Run() {
    Prepare();

    bool bQuit = false;
    while (!bQuit) {
        auto deltaTime = static_cast<float>(mFrameTimer.Frame());

        mWindow->PollEvents(
            bQuit, mStopRendering,
            [&](SDL_Event* e) { PollEvents(e, deltaTime); },
            [&]() { Update_OnResize(); });

        if (mStopRendering) {
            SDL_Delay(100);
        } else {
            UpdateScene();

            auto& frame = GetCurFrame();
            BeginFrame(frame);
            RenderFrame(frame);
            EndFrame(frame);
        }
    }
    mVulkanContext->GetDevice()->waitIdle();
}

void Application::RenderFrame(Core::RenderFrame& frame) {}

UniquePtr<SDLWindow> Application::CreateSDLWindow(const char* name,
                                                  uint32_t width,
                                                  uint32_t height) {
    return MakeUnique<SDLWindow>(name, width, height);
}

UniquePtr<VulkanContext> Application::CreateContext() {
    Type_STLVector<Type_STLString> requestedInstanceLayers {};
#ifndef NDEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    auto sdlRequestedInstanceExtensions = mWindow->GetVulkanInstanceExtension();
    Type_STLVector<Type_STLString> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifndef NDEBUG
    requestedInstanceExtensions.emplace_back(vk::EXTDebugUtilsExtensionName);
#endif
    requestedInstanceExtensions.emplace_back(
        vk::KHRGetPhysicalDeviceProperties2ExtensionName);

    Type_STLVector<Type_STLString> enabledDeivceExtensions {};

    enabledDeivceExtensions.emplace_back(vk::KHRSwapchainExtensionName);
    // enabledDeivceExtensions.emplace_back(
    //     vk::EXTDeviceAddressBindingReportExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTDescriptorBufferExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMeshShaderExtensionName);
    enabledDeivceExtensions.emplace_back(vk::KHRMaintenance5ExtensionName);
    enabledDeivceExtensions.emplace_back(vk::KHRMaintenance6ExtensionName);
    enabledDeivceExtensions.emplace_back(
        vk::KHRBufferDeviceAddressExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMemoryBudgetExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMemoryPriorityExtensionName);
    enabledDeivceExtensions.emplace_back(vk::KHRBindMemory2ExtensionName);
    enabledDeivceExtensions.emplace_back(
        vk::EXTDeviceGeneratedCommandsExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTShaderObjectExtensionName);

#ifdef CUDA_VULKAN_INTEROP
    enabledDeivceExtensions.emplace_back(
        vk::KHRExternalMemoryWin32ExtensionName);
    enabledDeivceExtensions.emplace_back(
        vk::KHRExternalSemaphoreWin32ExtensionName);
#endif

    return MakeUnique<VulkanContext>(*mWindow, requestedInstanceLayers,
                                     requestedInstanceExtensions,
                                     enabledDeivceExtensions);
}

UniquePtr<Swapchain> Application::CreateSwapchain() {
    return MakeUnique<Swapchain>(
        *mVulkanContext, vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mWindow->GetWidth()),
                      static_cast<uint32_t>(mWindow->GetHeight())});
}

UniquePtr<RenderResourceManager> Application::CreateRenderResourceManager() {
    return MakeUnique<RenderResourceManager>(*mVulkanContext);
}

UniquePtr<CommandManager> Application::CreateCommandManager() {
    return MakeUnique<CommandManager>(*mVulkanContext);
}

UniquePtr<PipelineManager> Application::CreatePipelineManager() {
    return MakeUnique<PipelineManager>(*mVulkanContext);
}

UniquePtr<ShaderManager> Application::CreateShaderManager() {
    return MakeUnique<ShaderManager>(*mVulkanContext);
}

UniquePtr<DGCSeqManager> Application::CreateDGCSeqManager() {
    return MakeUnique<DGCSeqManager>(*mVulkanContext, *mPipelineMgr,
                                     *mShaderMgr, *mRenderResMgr);
}

void Application::CreatePipelines() {}

void Application::LoadShaders() {}

void Application::UpdateScene() {}

void Application::Prepare() {
    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        mFrames.emplace_back(*mVulkanContext, *mRenderResMgr, i);
    }
}

void Application::BeginFrame(Core::RenderFrame& frame) {
    auto swapchainImageIdx = mSwapchain->AcquireNextImageIndex(frame);
}

void Application::EndFrame(Core::RenderFrame& frame) {
    auto& swapchain = GetSwapchain();
    auto extent = swapchain.GetExtent2D();
    // submit to swapchain
    {
        auto cmd = frame.GetGraphicsCmdBuf();
        auto scImgIdx = swapchain.GetCurrentImageIndex();
        auto imageHandle = swapchain.GetImageHandle(scImgIdx);
        auto viewHandle = swapchain.GetImageViewHandle(scImgIdx);
        vk::ImageMemoryBarrier2 preBarrier {};
        preBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
            .setSrcAccessMask(vk::AccessFlagBits2::eNone)
            .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSubresourceRange(Utils::GetWholeImageSubresource(
                vk::ImageAspectFlagBits::eColor))
            .setImage(imageHandle);

        vk::DependencyInfo preDep {};
        preDep.setImageMemoryBarriers(preBarrier);

        cmd.GetHandle().pipelineBarrier2(preDep);

        RenderToSwapchainBindings(cmd.GetHandle());

        vk::RenderingAttachmentInfo attachment {};
        attachment.setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setImageView(viewHandle);

        vk::RenderingInfo renderingInfo {};
        renderingInfo.setLayerCount(1)
            .setColorAttachments(attachment)
            .setRenderArea({{}, extent});

        cmd.GetHandle().beginRendering(renderingInfo);

        vk::Viewport viewport {
            0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
        cmd.GetHandle().setViewport(0, viewport);

        vk::Rect2D scissor {{0, 0}, {extent.width, extent.height}};
        cmd.GetHandle().setScissor(0, scissor);

        cmd.GetHandle().draw(3, 1, 0, 0);

        cmd.GetHandle().endRendering();

        vk::ImageMemoryBarrier2 postBarrier {};
        postBarrier
            .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
            .setDstAccessMask(vk::AccessFlagBits2::eNone)
            .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSubresourceRange(Utils::GetWholeImageSubresource(
                vk::ImageAspectFlagBits::eColor))
            .setImage(imageHandle);

        vk::DependencyInfo postDep {};
        postDep.setImageMemoryBarriers(postBarrier);

        cmd.GetHandle().pipelineBarrier2(postDep);

        mGui.Draw(cmd.GetHandle());

        cmd.End();

        Type_STLVector<SemSubmitInfo> waits = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             frame.GetRenderFinishedSemaphore().GetHandle()}};

        Type_STLVector<SemSubmitInfo> signals = {
            {vk::PipelineStageFlagBits2::eAllGraphics,
             frame.GetSwapchainPresentSemaphore().GetHandle()}};

        GetCmdMgr().Submit(
            cmd.GetHandle(),
            GetVulkanContext().GetQueue(QueueType::Graphics).GetHandle(), waits,
            signals, frame.GetFencePool().RequestFence());
    }

    mSwapchain->Present(
        frame, mVulkanContext->GetQueue(QueueType::Present).GetHandle());
    ++mFrameNum;
}

SDLWindow& Application::GetSDLWindow() const {
    return *mWindow;
}

VulkanContext& Application::GetVulkanContext() const {
    return *mVulkanContext;
}

Swapchain& Application::GetSwapchain() const {
    return *mSwapchain;
}

RenderResourceManager& Application::GetRenderResMgr() const {
    return *mRenderResMgr;
}

CommandManager& Application::GetCmdMgr() const {
    return *mCmdMgr;
}

PipelineManager& Application::GetPipelineMgr() const {
    return *mPipelineMgr;
}

ShaderManager& Application::GetShaderMgr() const {
    return *mShaderMgr;
}

DGCSeqManager& Application::GetDGCSeqMgr() const {
    return *mDGCSequenceMgr;
}

GUI& Application::GetUILayer() {
    return mGui;
}

RenderFrame& Application::GetCurFrame() {
    return mFrames[mFrameNum % FRAME_OVERLAP];
}

Type_STLVector<Core::RenderFrame> const& Application::GetFrames() const {
    return mFrames;
}

Type_STLVector<Core::RenderFrame>& Application::GetFrames() {
    return mFrames;
}

void Application::PollEvents(SDL_Event* e, float deltaTime) {}

void Application::Update_OnResize() {
    mSwapchain->Resize({static_cast<uint32_t>(mWindow->GetWidth()),
                        static_cast<uint32_t>(mWindow->GetHeight())});
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> Application::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mWindow->GetWidth()),
                                  static_cast<uint32_t>(mWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return mContext->CreateExternalImage2D(
        drawImageExtent, vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

void Application::CreateExternalTriangleData() {
    mTriangleExternalMesh.mVertexBuffer =
        mContext->CreateExternalPersistentBuffer(
            3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mTriangleExternalMesh.mIndexBuffer =
        mContext->CreateExternalPersistentBuffer(
            3 * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mContext->GetDevice()->getBufferAddress(deviceAddrInfo);
}

void Application::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mContext->GetDevice().GetHandle());
    mCUDASignalSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mContext->GetDevice().GetHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void Application::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

}  // namespace IntelliDesign_NS::Vulkan::Core