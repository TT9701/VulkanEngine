#include "Engine.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanHelper.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanSampler.hpp"
#include "VulkanSwapchain.hpp"
#include "Window.hpp"

VulkanEngine::VulkanEngine()
    : mSPWindow(CreateSDLWindow()),
      mSPContext(CreateContext()),
      mSPSwapchain(CreateSwapchain()),
      mDrawImage(CreateDrawImage()),
      mCUDAExternalImage(CreateExternalImage()),
      mSPCmdManager(CreateCommandManager()),
      mSPImmediateSubmitManager(CreateImmediateSubmitManager()),
      mErrorCheckImage(CreateErrorCheckTexture()) {
    SetCudaInterop();
    CreateCUDASyncStructures();

    CreateDefaultSamplers();

    CreateDescriptors();
    CreatePipelines();

    CreateTriangleData();
    CreateExternalTriangleData();
}

VulkanEngine::~VulkanEngine() {
    mSPContext->GetDeviceHandle().waitIdle();
}

void VulkanEngine::Run() {
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
    auto swapchainImage =
        mSPSwapchain->GetImageHandle(mSPSwapchain->AcquireNextImageIndex());

    auto cmd = mSPCmdManager->GetCmdBufferToBegin();

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eGeneral);

    DrawBackground(cmd);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eColorAttachmentOptimal);

    DrawTriangle(cmd);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);

    Utils::TransitionImageLayout(cmd, swapchainImage,
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferDstOptimal);

    mDrawImage->CopyToImage(
        cmd, swapchainImage,
        {mDrawImage->GetExtent3D().width, mDrawImage->GetExtent3D().height},
        mSPSwapchain->GetExtent2D());

    Utils::TransitionImageLayout(cmd, swapchainImage,
                                 vk::ImageLayout::eTransferDstOptimal,
                                 vk::ImageLayout::ePresentSrcKHR);

    mSPCmdManager->EndCmdBuffer(cmd);

    auto cmdInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);

    ::std::vector<vk::SemaphoreSubmitInfo> waitInfos {};
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        mSPSwapchain->GetReady4RenderSemHandle()));
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDASignalSemaphore->GetVkSemaphore()));

    ::std::vector<vk::SemaphoreSubmitInfo> signalInfos {};
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllGraphics,
        mSPSwapchain->GetReady4PresentSemHandle()));
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDAWaitSemaphore->GetVkSemaphore()));

    auto submit = Utils::SubmitInfo(cmdInfo, signalInfos, waitInfos);

    mSPCmdManager->Submit(mSPContext->GetDevice()->GetGraphicQueue(), submit);
    mSPCmdManager->GoToNextCmdBuffer();

    mSPSwapchain->Present(mSPContext->GetDevice()->GetGraphicQueue());

    // TODO: SYNCHRONIZATION
    mSPContext->GetDevice()->GetGraphicQueue().submit2(
        {}, mSPSwapchain->GetAquireFenceHandle());

    cudaExternalSemaphoreWaitParams waitParams {};
    auto cudaWait = mCUDAWaitSemaphore->GetCUDAExternalSemaphore();
    mCUDAStream.WaitExternalSemaphoresAsync(&cudaWait, &waitParams, 1);

    CUDA::SimPoint(mTriangleExternalMesh.mVertexBuffer
                       ->GetMappedPointer(0, 3 * sizeof(Vertex))
                       .GetPtr(),
                   mFrameNum, mCUDAStream.GetHandle());

    CUDA::SimSurface(*mCUDAExternalImage->GetSurfaceObjectPtr(), mFrameNum,
                     mCUDAStream.GetHandle());

    cudaExternalSemaphoreSignalParams signalParams {};
    auto cudaSignal = mCUDASignalSemaphore->GetCUDAExternalSemaphore();
    mCUDAStream.SignalExternalSemaphoresAsyn(&cudaSignal, &signalParams, 1);

    ++mFrameNum;
}

SharedPtr<SDLWindow> VulkanEngine::CreateSDLWindow() {
    return MakeShared<SDLWindow>();
}

SharedPtr<VulkanContext> VulkanEngine::CreateContext() {
    ::std::vector<::std::string> requestedInstanceLayers {};
#ifndef NDEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    auto sdlRequestedInstanceExtensions =
        mSPWindow->GetVulkanInstanceExtension();
    ::std::vector<::std::string> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifndef NDEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    ::std::vector<::std::string> enabledDeivceExtensions {};

    enabledDeivceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.emplace_back(
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);

    VulkanContext::EnableDefaultFeatures();

    return MakeShared<VulkanContext>(
        mSPWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

SharedPtr<VulkanSwapchain> VulkanEngine::CreateSwapchain() {
    return MakeShared<VulkanSwapchain>(
        mSPContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mSPWindow->GetWidth()),
                      static_cast<uint32_t>(mSPWindow->GetHeight())});
}

UniquePtr<VulkanAllocatedImage> VulkanEngine::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return MakeUnique<VulkanAllocatedImage>(
        mSPContext.get(), VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        drawImageExtent, vk::Format::eR16G16B16A16Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

UniquePtr<CUDA::VulkanExternalImage> VulkanEngine::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return MakeUnique<CUDA::VulkanExternalImage>(
        mSPContext->GetDeviceHandle(),
        mSPContext->GetVmaAllocator()->GetHandle(),
        mSPContext->GetExternalMemoryPool()->GetHandle(), 0, drawImageExtent,
        vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

SharedPtr<ImmediateSubmitManager> VulkanEngine::CreateImmediateSubmitManager() {
    return MakeShared<ImmediateSubmitManager>(
        mSPContext.get(),
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

SharedPtr<VulkanCommandManager> VulkanEngine::CreateCommandManager() {
    return MakeShared<VulkanCommandManager>(
        mSPContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

void VulkanEngine::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());
    mCUDASignalSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());

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

    mMainDescriptorAllocator.InitPool(mSPContext->GetDeviceHandle(), 10, sizes);

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
    mTriangleExternalMesh.mVertexBuffer =
        MakeUnique<CUDA::VulkanExternalBuffer>(
            mSPContext->GetDeviceHandle(),
            mSPContext->GetVmaAllocator()->GetHandle(), 3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            mSPContext->GetExternalMemoryPool()->GetHandle());

    mTriangleExternalMesh.mIndexBuffer = MakeUnique<CUDA::VulkanExternalBuffer>(
        mSPContext->GetDeviceHandle(),
        mSPContext->GetVmaAllocator()->GetHandle(), 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eIndexBuffer
            | vk::BufferUsageFlagBits::eTransferDst,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mSPContext->GetExternalMemoryPool()->GetHandle());

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    CUDA::VulkanExternalImage externalImage {
        mSPContext->GetDeviceHandle(),
        mSPContext->GetVmaAllocator()->GetHandle(),
        mSPContext->GetExternalMemoryPool()->GetHandle(),
        0,
        {16, 16, 1},
        vk::Format::eR8G8B8A8Uint,
        vk::ImageUsageFlagBits::eStorage,
        vk::ImageAspectFlagBits::eColor};

    auto cudaMipmapped = externalImage.GetMapMipmappedArray(0, 1);
}

UniquePtr<VulkanAllocatedImage> VulkanEngine::CreateErrorCheckTexture() {
    uint32_t black   = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    return MakeUnique<VulkanAllocatedImage>(
        mSPContext.get(), VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VkExtent3D {16, 16, 1}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor, pixels.data(), this);
}

void VulkanEngine::CreateDefaultSamplers() {
    mDefaultSamplerNearest = MakeShared<VulkanSampler>(
        mSPContext.get(), vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear = MakeShared<VulkanSampler>(
        mSPContext.get(), vk::Filter::eLinear, vk::Filter::eLinear);
}

void VulkanEngine::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mSPContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}

GPUMeshBuffers VulkanEngine::UploadMeshData(std::span<uint32_t> indices,
                                            std::span<Vertex>   vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize  = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newMesh {};
    newMesh.mVertexBuffer = MakeUnique<VulkanAllocatedBuffer>(
        mSPContext->GetVmaAllocator(), vertexBufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    newMesh.mIndexBuffer = MakeUnique<VulkanAllocatedBuffer>(
        mSPContext->GetVmaAllocator(), indexBufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer
            | vk::BufferUsageFlagBits::eTransferDst,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(newMesh.mVertexBuffer->GetHandle());

    newMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    VulkanAllocatedBuffer staging {
        mSPContext->GetVmaAllocator(), vertexBufferSize + indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT};

    void* data = staging.GetAllocationInfo().pMappedData;
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    mSPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
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

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    DescriptorLayoutBuilder builder;
    builder.AddBinding(0, vk::DescriptorType::eStorageImage);
    mDrawImageDescriptorLayout = builder.Build(
        mSPContext->GetDeviceHandle(), vk::ShaderStageFlagBits::eCompute);

    mDrawImageDescriptors = mMainDescriptorAllocator.Allocate(
        mSPContext->GetDeviceHandle(), mDrawImageDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(0,
                      {VK_NULL_HANDLE, mDrawImage->GetViewHandle(),
                       vk::ImageLayout::eGeneral},
                      vk::DescriptorType::eStorageImage);

    writer.UpdateSet(mSPContext->GetDeviceHandle(), mDrawImageDescriptors);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void VulkanEngine::CreateBackgroundComputePipeline() {
    vk::PipelineLayoutCreateInfo computeLayout {};
    computeLayout.setSetLayouts(mDrawImageDescriptorLayout);

    mBackgroundComputePipelineLayout =
        mSPContext->GetDeviceHandle().createPipelineLayout(computeLayout);

    vk::ShaderModule computeDrawShader {};
    VE_ASSERT(Utils::LoadShaderModule("../../Shaders/BackGround.comp.spv",
                                      mSPContext->GetDeviceHandle(),
                                      &computeDrawShader),
              "Error when building the compute shader");

    vk::PipelineShaderStageCreateInfo stageinfo {};
    stageinfo.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(computeDrawShader)
        .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo {};
    computePipelineCreateInfo.setLayout(mBackgroundComputePipelineLayout)
        .setStage(stageinfo);

    mBackgroundComputePipeline =
        mSPContext->GetDeviceHandle()
            .createComputePipeline({}, computePipelineCreateInfo)
            .value;

    mSPContext->GetDeviceHandle().destroy(computeDrawShader);

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void VulkanEngine::CreateTrianglePipeline() {
    vk::ShaderModule vertexShader {};
    vk::ShaderModule fragmentShader {};

    VE_ASSERT(
        Utils::LoadShaderModule("../../Shaders/Triangle.vert.spv",
                                mSPContext->GetDeviceHandle(), &vertexShader),
        "Error when building the triangle vertex shader");
    VE_ASSERT(
        Utils::LoadShaderModule("../../Shaders/Triangle.frag.spv",
                                mSPContext->GetDeviceHandle(), &fragmentShader),
        "Error when building the triangle fragment shader");

    vk::PushConstantRange pushConstant {};
    pushConstant.setSize(sizeof(MeshPushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    vk::PipelineLayoutCreateInfo layoutCreateInfo {};
    layoutCreateInfo.setPushConstantRanges(pushConstant)
        .setSetLayouts(mTextureTriangleDescriptorLayout);
    mTrianglePipelieLayout =
        mSPContext->GetDeviceHandle().createPipelineLayout(layoutCreateInfo);

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
            .Build(mSPContext->GetDeviceHandle());

    mSPContext->GetDeviceHandle().destroy(vertexShader);
    mSPContext->GetDeviceHandle().destroy(fragmentShader);

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void VulkanEngine::CreateTriangleDescriptors() {
    DescriptorLayoutBuilder builder {};
    builder.AddBinding(0, vk::DescriptorType::eCombinedImageSampler);
    mTextureTriangleDescriptorLayout = builder.Build(
        mSPContext->GetDeviceHandle(), vk::ShaderStageFlagBits::eFragment);

    mTextureTriangleDescriptors = mMainDescriptorAllocator.Allocate(
        mSPContext->GetDeviceHandle(), mTextureTriangleDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(
        0,
        {mDefaultSamplerNearest->GetHandle(), mErrorCheckImage->GetViewHandle(),
         vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    writer.UpdateSet(mSPContext->GetDeviceHandle(),
                     mTextureTriangleDescriptors);
}

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};
    float               flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));
    clearValue                = {flash, flash, flash, 1.0f};

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
    colorAttachment.setImageView(mDrawImage->GetViewHandle())
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
