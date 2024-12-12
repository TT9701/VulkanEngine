#include "Application.h"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace IntelliDesign_NS::Vulkan::Core {

Application::Application(ApplicationSpecification const& spec)
    : mWindow(CreateSDLWindow(spec.Name.c_str(), spec.width, spec.height)),
      mContext(CreateContext()),
      mSwapchain(CreateSwapchain()),
      mRenderResMgr(CreateRenderResourceManager()),
      mCmdMgr(CreateCommandManager()),
      mPipelineMgr(CreatePipelineManager()),
      mShaderMgr(CreateShaderManager())
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
    mContext->GetDevice()->waitIdle();
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
    mContext->GetDevice()->waitIdle();
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
    enabledDeivceExtensions.emplace_back(vk::EXTDescriptorBufferExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMeshShaderExtensionName);
    enabledDeivceExtensions.emplace_back(vk::KHRMaintenance6ExtensionName);
    enabledDeivceExtensions.emplace_back(
        vk::KHRBufferDeviceAddressExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMemoryBudgetExtensionName);
    enabledDeivceExtensions.emplace_back(vk::EXTMemoryPriorityExtensionName);
    enabledDeivceExtensions.emplace_back(vk::KHRBindMemory2ExtensionName);

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
        mContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mWindow->GetWidth()),
                      static_cast<uint32_t>(mWindow->GetHeight())});
}

RenderResourceManager Application::CreateRenderResourceManager() {
    return {&mContext->GetDevice(), mContext->GetVmaAllocator()};
}

CommandManager Application::CreateCommandManager() {
    return {mContext.get()};
}

PipelineManager Application::CreatePipelineManager() {
    return {mContext.get()};
}

ShaderManager Application::CreateShaderManager() {
    return {mContext.get()};
}

void Application::CreatePipelines() {}

void Application::LoadShaders() {}

void Application::UpdateScene() {}

void Application::Prepare() {
    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        mFrames.emplace_back(mContext.get());
    }
}

void Application::BeginFrame(Core::RenderFrame& frame) {
    auto swapchainImageIdx = mSwapchain->AcquireNextImageIndex(frame);
}

void Application::EndFrame(Core::RenderFrame& frame) {
    mSwapchain->Present(frame,
                        mContext->GetQueue(QueueUsage::Present).GetHandle());
    ++mFrameNum;
}

RenderFrame& Application::GetCurFrame() {
    return mFrames[mFrameNum % FRAME_OVERLAP];
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