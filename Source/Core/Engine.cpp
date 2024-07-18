#include "Engine.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include "Core/Utilities/VulkanUtilities.hpp"
#include "VulkanDebugUtils.hpp"
#include "VulkanDevice.hpp"
#include "VulkanImage.hpp"
#include "VulkanInstance.hpp"
#include "VulkanMemoryAllocator.hpp"
#include "VulkanPhysicalDevice.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanSurface.hpp"
#include "VulkanSwapchain.hpp"
#include "Window.hpp"

VulkanEngine::VulkanEngine()
    : mPMemPool(CreateGlobalMemoryPool()),
      mSPWindow(CreateSDLWindow()),
      mSPInstance(CreateInstance()),
      mPDebugUtilsMessenger(CreateDebugUtilsMessenger()),
      mSPSurface(CreateSurface()),
      mSPPhysicalDevice(PickPhysicalDevice()),
      mSPDevice(CreateDevice()),
      mSPVmaAllocator(CreateVmaAllocator()),
      mVmaExternalMemoryPool(CreateVmaExternalMemoryPool()),
      mDrawImage(CreateDrawImage()),
      mCUDAExternalImage(CreateExternalImage()),
      mSwapchain(CreateSwapchain()),
      mErrorCheckImage(CreateErrorCheckTexture()) {
    SetCudaInterop();
}

VulkanEngine::~VulkanEngine() {
    mSPDevice->GetHandle().waitIdle();

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mSPDevice->GetHandle().destroy(mFrameDatas[i].mCommandPool);
        mSPDevice->GetHandle().destroy(mFrameDatas[i].mRenderFence);
        mSPDevice->GetHandle().destroy(mFrameDatas[i].mReady4PresentSemaphore);
        mSPDevice->GetHandle().destroy(mFrameDatas[i].mReady4RenderSemaphore);
    }
}

void VulkanEngine::Init() {

    InitVulkan();
}

void VulkanEngine::Run() {
    SDL_Event sdlEvent;

    bool bQuit = false;

    while (!bQuit) {
        mSPWindow->PollEvents(bQuit, mStopRendering);

        if (mStopRendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            Draw();
        }
    }
}

void VulkanEngine::Draw() {
    VK_CHECK(mSPDevice->GetHandle().waitForFences(
        GetCurrentFrameData().mRenderFence, vk::True, TIME_OUT_NANO_SECONDS));

    mSPDevice->GetHandle().resetFences(GetCurrentFrameData().mRenderFence);

    uint32_t swapchainImageIndex;
    VK_CHECK(mSPDevice->GetHandle().acquireNextImageKHR(
        mSwapchain->GetHandle(), TIME_OUT_NANO_SECONDS,
        GetCurrentFrameData().mReady4RenderSemaphore, VK_NULL_HANDLE,
        &swapchainImageIndex));

    auto cmd = GetCurrentFrameData().mCommandBuffer;

    cmd.reset();

    vk::CommandBufferBeginInfo cmdBeginInfo {
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    cmd.begin(cmdBeginInfo);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eGeneral);

    DrawBackground(cmd);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eColorAttachmentOptimal);

    DrawTriangle(cmd);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);

    Utils::TransitionImageLayout(
        cmd, mSwapchain->GetImages()[swapchainImageIndex],
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    mDrawImage->CopyToImage(
        cmd, mSwapchain->GetImages()[swapchainImageIndex],
        {mDrawImage->GetExtent3D().width, mDrawImage->GetExtent3D().height},
        mSwapchain->GetExtent2D());

    Utils::TransitionImageLayout(
        cmd, mSwapchain->GetImages()[swapchainImageIndex],
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

    cmd.end();

    auto cmdInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);

    ::std::vector<vk::SemaphoreSubmitInfo> waitInfos {};
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        GetCurrentFrameData().mReady4RenderSemaphore));
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDASignalSemaphore.GetVkSemaphore()));

    ::std::vector<vk::SemaphoreSubmitInfo> signalInfos {};
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllGraphics,
        GetCurrentFrameData().mReady4PresentSemaphore));
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDAWaitSemaphore.GetVkSemaphore()));

    auto submit = Utils::SubmitInfo(cmdInfo, signalInfos, waitInfos);

    mSPDevice->GetGraphicQueues()[0].submit2(
        submit, GetCurrentFrameData().mRenderFence);

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain->GetHandle())
        .setWaitSemaphores(GetCurrentFrameData().mReady4PresentSemaphore)
        .setImageIndices(swapchainImageIndex);

    VK_CHECK(mSPDevice->GetGraphicQueues()[0].presentKHR(presentInfo));

    cudaExternalSemaphoreWaitParams waitParams {};
    mCUDAStream.WaitExternalSemaphoresAsync(
        &mCUDAWaitSemaphore.GetCUDAExternalSemaphore(), &waitParams, 1);

    CUDA::SimPoint(mTriangleExternalMesh.mVertexBuffer
                       .GetMappedPointer(0, 3 * sizeof(Vertex))
                       .GetPtr(),
                   mFrameNum, mCUDAStream.Get());

    CUDA::SimSurface(*mCUDAExternalImage->GetSurfaceObjectPtr(), mFrameNum,
                     mCUDAStream.Get());

    cudaExternalSemaphoreSignalParams signalParams {};
    mCUDAStream.SignalExternalSemaphoresAsyn(
        &mCUDASignalSemaphore.GetCUDAExternalSemaphore(), &signalParams, 1);

    ++mFrameNum;
}

void VulkanEngine::InitVulkan() {
    CreateCommands();
    CreateSyncStructures();

    CreateErrorCheckTexture();
    CreateDefaultSamplers();

    CreateDescriptors();
    CreatePipelines();

    CreateTriangleData();
    CreateExternalTriangleData();
}

std::pmr::memory_resource* VulkanEngine::CreateGlobalMemoryPool() {
    return ::std::pmr::get_default_resource();
}

VulkanEngine::Type_SPInstance<SDLWindow> VulkanEngine::CreateSDLWindow() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<SDLWindow>(mPMemPool);
}

VulkanEngine::Type_SPInstance<VulkanInstance> VulkanEngine::CreateInstance() {
    ::std::vector<::std::string> requestedInstanceLayers {};
#ifdef DEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    auto sdlRequestedInstanceExtensions =
        mSPWindow->GetVulkanInstanceExtension();
    ::std::vector<::std::string> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifdef DEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanInstance>(
        mPMemPool, requestedInstanceLayers, requestedInstanceExtensions);
}

#ifdef DEBUG
VulkanEngine::Type_PInstance<VulkanDebugUtils>
VulkanEngine::CreateDebugUtilsMessenger() {
    return IntelliDesign_NS::Core::MemoryPool::New_Unique<VulkanDebugUtils>(
        mPMemPool, mSPInstance);
}
#endif

VulkanEngine::Type_SPInstance<VulkanSurface> VulkanEngine::CreateSurface() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanSurface>(
        mPMemPool, mSPInstance, mSPWindow);
}

VulkanEngine::Type_SPInstance<VulkanPhysicalDevice>
VulkanEngine::PickPhysicalDevice() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanPhysicalDevice>(
        mPMemPool, mSPInstance,
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
}

VulkanEngine::Type_SPInstance<VulkanDevice> VulkanEngine::CreateDevice() {
    ::std::vector<::std::string> enabledDeivceExtensions {};

    enabledDeivceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);

    vk::PhysicalDeviceFeatures origFeatures {};

    vk::PhysicalDeviceVulkan13Features vulkan13Features {};
    vulkan13Features.setDynamicRendering(vk::True).setSynchronization2(
        vk::True);

    vk::PhysicalDeviceVulkan12Features vulkan12Features {};
    vulkan12Features.setBufferDeviceAddress(vk::True)
        .setDescriptorIndexing(vk::True)
        .setPNext(&vulkan13Features);

    vk::PhysicalDeviceVulkan11Features vulkan11Features {};
    vulkan11Features.setPNext(&vulkan12Features);

    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanDevice>(
        mPMemPool, mSPPhysicalDevice, ::std::vector<::std::string> {},
        enabledDeivceExtensions, &origFeatures, &vulkan11Features);
}

VulkanEngine::Type_SPInstance<VulkanMemoryAllocator>
VulkanEngine::CreateVmaAllocator() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<
        VulkanMemoryAllocator>(mPMemPool, mSPPhysicalDevice, mSPDevice,
                               mSPInstance);
}

VulkanEngine::Type_SPInstance<VulkanExternalMemoryPool>
VulkanEngine::CreateVmaExternalMemoryPool() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<
        VulkanExternalMemoryPool>(mPMemPool, mSPVmaAllocator);
}

VulkanEngine::Type_SPInstance<VulkanSwapchain> VulkanEngine::CreateSwapchain() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanSwapchain>(
        mPMemPool, mSPDevice, mSPSurface, vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mSPWindow->GetWidth()),
                      static_cast<uint32_t>(mSPWindow->GetHeight())});
}

VulkanEngine::Type_PInstance<VulkanAllocatedImage>
VulkanEngine::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return IntelliDesign_NS::Core::MemoryPool::New_Unique<VulkanAllocatedImage>(
        mPMemPool, mSPDevice, mSPVmaAllocator,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, drawImageExtent,
        vk::Format::eR16G16B16A16Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

VulkanEngine::Type_PInstance<CUDA::VulkanExternalImage>
VulkanEngine::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    auto temp_externalImage = IntelliDesign_NS::Core::MemoryPool::New_Unique<
        CUDA::VulkanExternalImage>(mPMemPool);

    temp_externalImage->CreateExternalImage(
        mSPDevice->GetHandle(), mSPVmaAllocator->GetHandle(),
        mVmaExternalMemoryPool->GetHandle(), 0, drawImageExtent,
        vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);

    return temp_externalImage;
}

void VulkanEngine::CreateCommands() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(
            mSPPhysicalDevice->GetGraphicsQueueFamilyIndex().value());

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mFrameDatas[i].mCommandPool =
            mSPDevice->GetHandle().createCommandPool(cmdPoolCreateInfo);

        vk::CommandBufferAllocateInfo cmdAllocInfo {};
        cmdAllocInfo.setCommandPool(mFrameDatas[i].mCommandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1u);
        mFrameDatas[i].mCommandBuffer =
            mSPDevice->GetHandle().allocateCommandBuffers(cmdAllocInfo)[0];
    }

    DBG_LOG_INFO("Vulkan Per Frame CommandPool & CommandBuffer Created");

    mImmediateSubmit.mCommandPool =
        mSPDevice->GetHandle().createCommandPool(cmdPoolCreateInfo);

    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(mImmediateSubmit.mCommandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1u);
    mImmediateSubmit.mCommandBuffer =
        mSPDevice->GetHandle().allocateCommandBuffers(cmdAllocInfo)[0];

    DBG_LOG_INFO("Vulkan Immediate submit CommandPool & CommandBuffer Created");
}

void VulkanEngine::CreateSyncStructures() {
    vk::FenceCreateInfo fenceCreateInfo {vk::FenceCreateFlagBits::eSignaled};
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mFrameDatas[i].mRenderFence =
            mSPDevice->GetHandle().createFence(fenceCreateInfo);
        mFrameDatas[i].mReady4PresentSemaphore =
            mSPDevice->GetHandle().createSemaphore(semaphoreCreateInfo);
        mFrameDatas[i].mReady4RenderSemaphore =
            mSPDevice->GetHandle().createSemaphore(semaphoreCreateInfo);
    }

    DBG_LOG_INFO("Vulkan Per Frame Fence & Semaphore Created");

    mImmediateSubmit.mFence =
        mSPDevice->GetHandle().createFence(fenceCreateInfo);

    DBG_LOG_INFO("Vulkan Immediate submit Fence & Semaphore Created");

    mCUDAWaitSemaphore.CreateExternalSemaphore(mSPDevice->GetHandle());
    mCUDASignalSemaphore.CreateExternalSemaphore(mSPDevice->GetHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void VulkanEngine::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateTrianglePipeline();
}

void VulkanEngine::CreateDescriptors() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};

    mMainDescriptorAllocator.InitPool(mSPDevice->GetHandle(), 10, sizes);

    CreateBackgroundComputeDescriptors();
    CreateTriangleDescriptors();
}

void VulkanEngine::CreateTriangleData() {
    ::std::array<Vertex, 3> vertices {};

    vertices[0].position = {0.0f, 0.5f, 0.0f};
    vertices[1].position = {-1.0f, 0.5f, 0.0f};
    vertices[2].position = {-0.5f, -0.5f, 0.0f};

    vertices[0].color = {1.0f, 0.0f, 0.0f, 1.0f};
    vertices[1].color = {0.0f, 1.0f, 0.0f, 1.0f};
    vertices[2].color = {0.0f, 0.0f, 1.0f, 1.0f};

    vertices[0].uvX = 1.0f;
    vertices[1].uvX = 0.0f;
    vertices[2].uvX = 0.5f;

    vertices[0].uvY = 0.0f;
    vertices[1].uvY = 0.0f;
    vertices[2].uvY = 1.0f;

    ::std::array<uint32_t, 3> indices {0, 1, 2};

    mTriangleMesh = UploadMeshData(indices, vertices);
}

void VulkanEngine::CreateExternalTriangleData() {
    CUDA::VulkanExternalBuffer vertexBuffer {}, indexBuffer {};
    vertexBuffer.CreateExternalBuffer(
        mSPDevice->GetHandle(), mSPVmaAllocator->GetHandle(),
        3 * sizeof(Vertex),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mVmaExternalMemoryPool->GetHandle());

    indexBuffer.CreateExternalBuffer(
        mSPDevice->GetHandle(), mSPVmaAllocator->GetHandle(),
        3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eIndexBuffer
            | vk::BufferUsageFlagBits::eTransferDst,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mVmaExternalMemoryPool->GetHandle());

    mTriangleExternalMesh.mVertexBuffer = vertexBuffer;
    mTriangleExternalMesh.mIndexBuffer = indexBuffer;

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(mTriangleExternalMesh.mVertexBuffer.GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mSPDevice->GetHandle().getBufferAddress(deviceAddrInfo);

    CUDA::VulkanExternalImage externalImage {};
    externalImage.CreateExternalImage(
        mSPDevice->GetHandle(), mSPVmaAllocator->GetHandle(),
        mVmaExternalMemoryPool->GetHandle(), 0, {16, 16, 1},
        vk::Format::eR8G8B8A8Uint, vk::ImageUsageFlagBits::eStorage,
        vk::ImageAspectFlagBits::eColor);

    auto cudaMipmapped = externalImage.GetMapMipmappedArray(0, 1);

    externalImage.Destroy(mSPDevice->GetHandle(), mSPVmaAllocator->GetHandle());
}

VulkanEngine ::Type_PInstance<VulkanAllocatedImage>
VulkanEngine::CreateErrorCheckTexture() {
    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    return IntelliDesign_NS::Core::MemoryPool::New_Unique<VulkanAllocatedImage>(
        mPMemPool, mSPDevice, mSPVmaAllocator,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, VkExtent3D {16, 16, 1},
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor, pixels.data(), this);
}

void VulkanEngine::CreateDefaultSamplers() {
    vk::SamplerCreateInfo info {};
    info.setMinFilter(vk::Filter::eNearest).setMagFilter(vk::Filter::eNearest);
    mDefaultSamplerNearest = mSPDevice->GetHandle().createSampler(info);
    info.setMinFilter(vk::Filter::eLinear).setMagFilter(vk::Filter::eLinear);
    mDefaultSamplerLinear = mSPDevice->GetHandle().createSampler(info);
}

void VulkanEngine::SetCudaInterop() {
    auto result =
        CUDA::GetVulkanCUDABindDeviceID(mSPPhysicalDevice->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}

GPUMeshBuffers VulkanEngine::UploadMeshData(std::span<uint32_t> indices,
                                            std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newMesh {};
    newMesh.mVertexBuffer =
        IntelliDesign_NS::Core::MemoryPool::New_Unique<VulkanAllocatedBuffer>(
            mPMemPool, mSPVmaAllocator, vertexBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    newMesh.mIndexBuffer =
        IntelliDesign_NS::Core::MemoryPool::New_Unique<VulkanAllocatedBuffer>(
            mPMemPool, mSPVmaAllocator, indexBufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer
                | vk::BufferUsageFlagBits::eTransferDst,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(newMesh.mVertexBuffer->GetHandle());

    newMesh.mVertexBufferAddress =
        mSPDevice->GetHandle().getBufferAddress(deviceAddrInfo);

    VulkanAllocatedBuffer staging {
        mSPVmaAllocator, vertexBufferSize + indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT};

    void* data = staging.GetAllocationInfo().pMappedData;
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    ImmediateSubmit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging.GetHandle(), newMesh.mVertexBuffer->GetHandle(),
                       vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging.GetHandle(), newMesh.mIndexBuffer->GetHandle(),
                       indexCopy);
    });

    return newMesh;
}

void VulkanEngine::ImmediateSubmit(
    std::function<void(vk::CommandBuffer cmd)>&& function) {
    mSPDevice->GetHandle().resetFences(mImmediateSubmit.mFence);

    auto cmd = mImmediateSubmit.mCommandBuffer;

    cmd.reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);

    function(cmd);

    cmd.end();

    auto cmdSubmitInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);
    auto submit = Utils::SubmitInfo(cmdSubmitInfo, {}, {});

    mSPDevice->GetGraphicQueues()[0].submit2(submit, mImmediateSubmit.mFence);
    VK_CHECK(mSPDevice->GetHandle().waitForFences(
        mImmediateSubmit.mFence, vk::True, TIME_OUT_NANO_SECONDS));
}

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    DescriptorLayoutBuilder builder;
    builder.AddBinding(0, vk::DescriptorType::eStorageImage);
    mDrawImageDescriptorLayout = builder.Build(
        mSPDevice->GetHandle(), vk::ShaderStageFlagBits::eCompute);

    mDrawImageDescriptors = mMainDescriptorAllocator.Allocate(
        mSPDevice->GetHandle(), mDrawImageDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(
        0,
        {VK_NULL_HANDLE, mDrawImage->GetImageView(), vk::ImageLayout::eGeneral},
        vk::DescriptorType::eStorageImage);

    writer.UpdateSet(mSPDevice->GetHandle(), mDrawImageDescriptors);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void VulkanEngine::CreateBackgroundComputePipeline() {
    vk::PipelineLayoutCreateInfo computeLayout {};
    computeLayout.setSetLayouts(mDrawImageDescriptorLayout);

    mBackgroundComputePipelineLayout =
        mSPDevice->GetHandle().createPipelineLayout(computeLayout);

    vk::ShaderModule computeDrawShader {};
    assert(Utils::LoadShaderModule("../../Shaders/BackGround.comp.spv",
                                   mSPDevice->GetHandle(), &computeDrawShader),
           "Error when building the compute shader");

    vk::PipelineShaderStageCreateInfo stageinfo {};
    stageinfo.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(computeDrawShader)
        .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo {};
    computePipelineCreateInfo.setLayout(mBackgroundComputePipelineLayout)
        .setStage(stageinfo);

    mBackgroundComputePipeline =
        mSPDevice->GetHandle()
            .createComputePipeline({}, computePipelineCreateInfo)
            .value;

    mSPDevice->GetHandle().destroy(computeDrawShader);

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void VulkanEngine::CreateTrianglePipeline() {
    vk::ShaderModule vertexShader {};
    vk::ShaderModule fragmentShader {};

    assert(Utils::LoadShaderModule("../../Shaders/Triangle.vert.spv",
                                   mSPDevice->GetHandle(), &vertexShader),
           "Error when building the triangle vertex shader");
    assert(Utils::LoadShaderModule("../../Shaders/Triangle.frag.spv",
                                   mSPDevice->GetHandle(), &fragmentShader),
           "Error when building the triangle fragment shader");

    vk::PushConstantRange pushConstant {};
    pushConstant.setSize(sizeof(MeshPushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    vk::PipelineLayoutCreateInfo layoutCreateInfo {};
    layoutCreateInfo.setPushConstantRanges(pushConstant)
        .setSetLayouts(mTextureTriangleDescriptorLayout);
    mTrianglePipelieLayout =
        mSPDevice->GetHandle().createPipelineLayout(layoutCreateInfo);

    GraphicsPipelineBuilder graphicsPipelineBuilder {};
    mTrianglePipelie =
        graphicsPipelineBuilder.SetLayout(mTrianglePipelieLayout)
            .SetShaders(vertexShader, fragmentShader)
            .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
            .SetPolygonMode(vk::PolygonMode::eFill)
            .SetCullMode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise)
            .SetMultisampling(vk::SampleCountFlagBits::e1)
            .SetBlending(vk::False)
            .SetDepth(vk::False, vk::False)
            .SetColorAttachmentFormat(mDrawImage->GetFormat())
            .SetDepthStencilFormat(vk::Format::eUndefined)
            .Build(mSPDevice->GetHandle());

    mSPDevice->GetHandle().destroy(vertexShader);
    mSPDevice->GetHandle().destroy(fragmentShader);

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void VulkanEngine::CreateTriangleDescriptors() {
    DescriptorLayoutBuilder builder {};
    builder.AddBinding(0, vk::DescriptorType::eCombinedImageSampler);
    mTextureTriangleDescriptorLayout = builder.Build(
        mSPDevice->GetHandle(), vk::ShaderStageFlagBits::eFragment);

    mTextureTriangleDescriptors = mMainDescriptorAllocator.Allocate(
        mSPDevice->GetHandle(), mTextureTriangleDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(0,
                      {mDefaultSamplerNearest, mErrorCheckImage->GetImageView(),
                       vk::ImageLayout::eShaderReadOnlyOptimal},
                      vk::DescriptorType::eCombinedImageSampler);

    writer.UpdateSet(mSPDevice->GetHandle(), mTextureTriangleDescriptors);
}

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};
    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));
    clearValue = {flash, flash, flash, 1.0f};

    auto subresource =
        Utils::GetDefaultImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

    cmd.clearColorImage(mDrawImage->GetHandle(), vk::ImageLayout::eGeneral,
                        clearValue, subresource);

    // cmd.bindPipeline(vk::PipelineBindPoint::eCompute,
    //                  mBackgroundComputePipeline);
    //
    // cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
    //                        mBackgroundComputePipelineLayout, 0,
    //                        mDrawImageDescriptors, {});
    //
    // cmd.dispatch(::std::ceil(mDrawImage.mExtent3D.width / 16.0),
    //              ::std::ceil(mDrawImage.mExtent3D.height / 16.0), 1);

    auto layout = mDrawImage->GetLayout();

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferSrcOptimal);

    vk::ImageBlit2 blitRegion {};
    blitRegion
        .setSrcOffsets(
            {vk::Offset3D {},
             vk::Offset3D {
                 static_cast<int32_t>(mCUDAExternalImage->GetExtent3D().width),
                 static_cast<int32_t>(mCUDAExternalImage->GetExtent3D().height),
                 1}})
        .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setDstOffsets(
            {vk::Offset3D {},
             vk::Offset3D {
                 static_cast<int32_t>(mDrawImage->GetExtent3D().width),
                 static_cast<int32_t>(mDrawImage->GetExtent3D().height), 1}})
        .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});

    vk::BlitImageInfo2 blitInfo {};
    blitInfo.setDstImage(mDrawImage->GetHandle())
        .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcImage(mCUDAExternalImage->GetVkImage())
        .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setFilter(vk::Filter::eLinear)
        .setRegions(blitRegion);

    cmd.blitImage2(blitInfo);

    mDrawImage->TransitionLayout(cmd, layout);
    Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
                                 vk::ImageLayout::eTransferSrcOptimal,
                                 vk::ImageLayout::eGeneral);
}

void VulkanEngine::DrawTriangle(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mDrawImage->GetImageView())
        .setImageLayout(mDrawImage->GetLayout())
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {{0, 0},
                                   {mDrawImage->GetExtent3D().width,
                                    mDrawImage->GetExtent3D().height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(VK_NULL_HANDLE)     // TODO
        .setPStencilAttachment(VK_NULL_HANDLE);  // TODO

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mTrianglePipelie);

    vk::Viewport viewport {};
    viewport.setWidth(mDrawImage->GetExtent3D().width)
        .setHeight(mDrawImage->GetExtent3D().height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mDrawImage->GetExtent3D().width, mDrawImage->GetExtent3D().height});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           mTrianglePipelieLayout, 0,
                           mTextureTriangleDescriptors, {});

    MeshPushConstants pushConstants {};
    pushConstants.mVertexBufferAddress =
        mTriangleExternalMesh.mVertexBufferAddress;
    pushConstants.mModelMatrix = glm::mat4(1.0f);
    cmd.pushConstants(mTrianglePipelieLayout, vk::ShaderStageFlagBits::eVertex,
                      0, sizeof(MeshPushConstants), &pushConstants);

    cmd.bindIndexBuffer(mTriangleMesh.mIndexBuffer->GetHandle(), 0,
                        vk::IndexType::eUint32);

    cmd.drawIndexed(3, 1, 0, 0, 0);

    cmd.endRendering();
}
