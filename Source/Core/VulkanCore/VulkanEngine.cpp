#include "VulkanEngine.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include <glm/glm.hpp>

#include "Core/Platform/Window.hpp"
#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanDescriptors.hpp"
#include "VulkanHelper.hpp"
#include "VulkanImage.hpp"
#include "VulkanShader.hpp"
#include "VulkanSwapchain.hpp"

VulkanEngine::VulkanEngine()
    : mSPWindow(CreateSDLWindow()),
      mSPContext(CreateContext()),
      mSPSwapchain(CreateSwapchain()),
      mDrawImage(CreateDrawImage()),
#ifdef CUDA_VULKAN_INTEROP
      mCUDAExternalImage(CreateExternalImage()),
#endif
      mSPCmdManager(CreateCommandManager()),
      mSPImmediateSubmitManager(CreateImmediateSubmitManager()),
      mErrorCheckImage(CreateErrorCheckTexture()),
      mDescriptorManager(CreateDescriptorManager()),
      mPipelineManager(CreatePipelineManager()) {
    CreateDescriptors();
    CreatePipelines();

    CreateTriangleData();

#ifdef CUDA_VULKAN_INTEROP
    SetCudaInterop();
    CreateCUDASyncStructures();
    CreateExternalTriangleData();
#endif
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

    DrawMesh(cmd);

    mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Utils::TransitionImageLayout(cmd, swapchainImage,
    //                              vk::ImageLayout::eUndefined,
    //                              vk::ImageLayout::eTransferDstOptimal);
    //
    // mDrawImage->CopyToImage(
    //     cmd, swapchainImage,
    //     {mDrawImage->GetExtent3D().width, mDrawImage->GetExtent3D().height},
    //     mSPSwapchain->GetExtent2D());
    //
    // Utils::TransitionImageLayout(cmd, swapchainImage,
    //                              vk::ImageLayout::eTransferDstOptimal,
    //                              vk::ImageLayout::ePresentSrcKHR);

    Utils::TransitionImageLayout(cmd, swapchainImage,
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eColorAttachmentOptimal);

    DrawQuad(cmd);

    Utils::TransitionImageLayout(cmd, swapchainImage,
                                 vk::ImageLayout::eColorAttachmentOptimal,
                                 vk::ImageLayout::ePresentSrcKHR);

    mSPCmdManager->EndCmdBuffer(cmd);

    auto cmdInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);

    ::std::vector<vk::SemaphoreSubmitInfo> waitInfos {};
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        mSPSwapchain->GetReady4RenderSemHandle()));

#ifdef CUDA_VULKAN_INTEROP
    waitInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDASignalSemaphore->GetVkSemaphore()));
#endif

    ::std::vector<vk::SemaphoreSubmitInfo> signalInfos {};
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllGraphics,
        mSPSwapchain->GetReady4PresentSemHandle()));

#ifdef CUDA_VULKAN_INTEROP
    signalInfos.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllCommands,
        mCUDAWaitSemaphore->GetVkSemaphore()));
#endif

    auto submit = Utils::SubmitInfo(cmdInfo, signalInfos, waitInfos);

    mSPCmdManager->Submit(mSPContext->GetDevice()->GetGraphicQueue(), submit);
    mSPCmdManager->GoToNextCmdBuffer();

    mSPSwapchain->Present(mSPContext->GetDevice()->GetGraphicQueue());

#ifdef CUDA_VULKAN_INTEROP
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
#endif

    ++mFrameNum;
}

UniquePtr<SDLWindow> VulkanEngine::CreateSDLWindow() {
    return MakeUnique<SDLWindow>();
}

UniquePtr<VulkanContext> VulkanEngine::CreateContext() {
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

    return MakeUnique<VulkanContext>(
        mSPWindow.get(),
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
        requestedInstanceLayers, requestedInstanceExtensions,
        enabledDeivceExtensions);
}

UniquePtr<VulkanSwapchain> VulkanEngine::CreateSwapchain() {
    return MakeUnique<VulkanSwapchain>(
        mSPContext.get(), vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D {static_cast<uint32_t>(mSPWindow->GetWidth()),
                      static_cast<uint32_t>(mSPWindow->GetHeight())});
}

SharedPtr<VulkanImage> VulkanEngine::CreateDrawImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsage |= vk::ImageUsageFlagBits::eSampled;

    return mSPContext->CreateImage2D(
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, drawImageExtent,
        vk::Format::eR16G16B16A16Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

UniquePtr<ImmediateSubmitManager> VulkanEngine::CreateImmediateSubmitManager() {
    return MakeUnique<ImmediateSubmitManager>(
        mSPContext.get(),
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

UniquePtr<VulkanCommandManager> VulkanEngine::CreateCommandManager() {
    return MakeUnique<VulkanCommandManager>(
        mSPContext.get(), FRAME_OVERLAP, FRAME_OVERLAP,
        mSPContext->GetPhysicalDevice()->GetGraphicsQueueFamilyIndex().value());
}

void VulkanEngine::CreatePipelines() {
    CreateBackgroundComputePipeline();
    CreateMeshPipeline();
    CreateDrawQuadPipeline();
}

void VulkanEngine::CreateDescriptors() {

    CreateBackgroundComputeDescriptors();
    CreateMeshDescriptors();
    CreateDrawQuadDescriptors();
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

    mBoxMesh = UploadMeshData(indices, vertices);
}

SharedPtr<VulkanImage> VulkanEngine::CreateErrorCheckTexture() {
    uint32_t black   = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    return mSPContext->CreateImage2D(
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, VkExtent3D {16, 16, 1},
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor, pixels.data(), this);
}

UniquePtr<VulkanDescriptorManager> VulkanEngine::CreateDescriptorManager() {
    std::vector<DescPoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};

    return MakeUnique<VulkanDescriptorManager>(mSPContext.get(), 10, sizes);
}

UniquePtr<VulkanPipelineManager> VulkanEngine::CreatePipelineManager() {
    return MakeUnique<VulkanPipelineManager>(mSPContext.get());
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> VulkanEngine::CreateExternalImage() {
    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mSPWindow->GetWidth()),
                                  static_cast<uint32_t>(mSPWindow->GetHeight()),
                                  1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    return mSPContext->CreateExternalImage2D(
        drawImageExtent, vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);
}

void VulkanEngine::CreateExternalTriangleData() {
    mTriangleExternalMesh.mVertexBuffer =
        mSPContext->CreateExternalPersistentBuffer(
            3 * sizeof(Vertex),
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mTriangleExternalMesh.mIndexBuffer =
        mSPContext->CreateExternalPersistentBuffer(
            3 * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(
        mTriangleExternalMesh.mVertexBuffer->GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);
}

void VulkanEngine::CreateCUDASyncStructures() {
    mCUDAWaitSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());
    mCUDASignalSemaphore = MakeShared<CUDA::VulkanExternalSemaphore>(
        mSPContext->GetDeviceHandle());

    DBG_LOG_INFO("Vulkan CUDA External Semaphore Created");
}

void VulkanEngine::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(
        mSPContext->GetPhysicalDevice()->GetHandle());
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}
#endif

GPUMeshBuffers VulkanEngine::UploadMeshData(std::span<uint32_t> indices,
                                            std::span<Vertex>   vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize  = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newMesh {};
    newMesh.mVertexBuffer = mSPContext->CreatePersistentBuffer(
        vertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer
                              | vk::BufferUsageFlagBits::eTransferDst
                              | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    newMesh.mIndexBuffer = mSPContext->CreatePersistentBuffer(
        indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer
                             | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(newMesh.mVertexBuffer->GetHandle());

    newMesh.mVertexBufferAddress =
        mSPContext->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    auto staging =
        mSPContext->CreateStagingBuffer(vertexBufferSize + indexBufferSize,
                                        vk::BufferUsageFlagBits::eTransferSrc);

    void* data = staging->GetAllocationInfo().pMappedData;
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    mSPImmediateSubmitManager->Submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), newMesh.mVertexBuffer->GetHandle(),
                       vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), newMesh.mIndexBuffer->GetHandle(),
                       indexCopy);
    });

    return newMesh;
}

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eStorageImage);
    const auto drawImageSetLayout = mDescriptorManager->BuildDescSetLayout(
        "DrawImage_Layout_0", vk::ShaderStageFlagBits::eCompute);

    const auto drawImageDesc =
        mDescriptorManager->Allocate("DrawImage_Desc_0", drawImageSetLayout);

    mDescriptorManager->WriteImage(0,
                                   {VK_NULL_HANDLE, mDrawImage->GetViewHandle(),
                                    vk::ImageLayout::eGeneral},
                                   vk::DescriptorType::eStorageImage);

    mDescriptorManager->UpdateSet(drawImageDesc);

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void VulkanEngine::CreateBackgroundComputePipeline() {
    ::std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("DrawImage_Layout_0")};

    auto backgroundPipelineLayout = mPipelineManager->CreateLayout(
        "BackgoundCompute_Layout", setLayouts, {});

    VulkanShader computeDrawShader {mSPContext.get(), "computeDraw",
                                    "../../Shaders/BackGround.comp.spv",
                                    ShaderStage::Compute};

    auto& builder = mPipelineManager->GetComputePipelineBuilder();

    auto backgroundComputePipeline =
        builder.SetShader(computeDrawShader)
            .SetLayout(backgroundPipelineLayout->GetHandle())
            .Build("BackgroundCompute_Pipeline");

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void VulkanEngine::CreateMeshPipeline() {
    std::vector<VulkanShader> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mSPContext.get(), "vertex",
                         "../../Shaders/Triangle.vert.spv",
                         ShaderStage::Vertex);

    shaders.emplace_back(mSPContext.get(), "fragment",
                         "../../Shaders/Triangle.frag.spv",
                         ShaderStage::Fragment);

    vk::PushConstantRange pushConstant {};
    pushConstant.setSize(sizeof(MeshPushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("Triangle_Layout_0")};

    auto trianglePipelineLayout = mPipelineManager->CreateLayout(
        "Triangle_Layout", setLayouts, pushConstant);

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(trianglePipelineLayout->GetHandle())
        .SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::False, vk::False)
        .SetColorAttachmentFormat(mDrawImage->GetFormat())
        .SetDepthStencilFormat(vk::Format::eUndefined)
        .Build("TriangleDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void VulkanEngine::CreateMeshDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto triangleSetLayout = mDescriptorManager->BuildDescSetLayout(
        "Triangle_Layout_0", vk::ShaderStageFlagBits::eFragment);

    const auto triangleDesc =
        mDescriptorManager->Allocate("Triangle_Desc_0", triangleSetLayout);

    mDescriptorManager->WriteImage(
        0,
        {mSPContext->GetDefaultNearestSamplerHandle(),
         mErrorCheckImage->GetViewHandle(),
         vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(triangleDesc);
}

void VulkanEngine::CreateDrawQuadDescriptors() {
    mDescriptorManager->AddDescSetLayoutBinding(
        0, 1, vk::DescriptorType::eCombinedImageSampler);

    const auto quadSetLayout = mDescriptorManager->BuildDescSetLayout(
        "Quad_Layout_0", vk::ShaderStageFlagBits::eFragment);

    const auto quadDesc =
        mDescriptorManager->Allocate("Quad_Desc_0", quadSetLayout);

    mDescriptorManager->WriteImage(
        0,
        {mSPContext->GetDefaultLinearSamplerHandle(),
         mDrawImage->GetViewHandle(), vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorType::eCombinedImageSampler);

    mDescriptorManager->UpdateSet(quadDesc);
}

void VulkanEngine::CreateDrawQuadPipeline() {
    std::vector<VulkanShader> shaders;
    shaders.reserve(2);

    shaders.emplace_back(mSPContext.get(), "vertex",
                         "../../Shaders/Quad.vert.spv", ShaderStage::Vertex);

    shaders.emplace_back(mSPContext.get(), "fragment",
                         "../../Shaders/Quad.frag.spv", ShaderStage::Fragment);

    std::vector setLayouts {
        mDescriptorManager->GetDescSetLayout("Quad_Layout_0")};

    auto quadPipelineLayout =
        mPipelineManager->CreateLayout("Quad_Layout", setLayouts);

    auto& builder = mPipelineManager->GetGraphicsPipelineBuilder();
    builder.SetLayout(quadPipelineLayout->GetHandle())
        .SetShaders(shaders)
        .SetInputTopology(vk::PrimitiveTopology::eTriangleList)
        .SetPolygonMode(vk::PolygonMode::eFill)
        .SetCullMode(vk::CullModeFlagBits::eFront,
                     vk::FrontFace::eCounterClockwise)
        .SetMultisampling(vk::SampleCountFlagBits::e1)
        .SetBlending(vk::False)
        .SetDepth(vk::False, vk::False)
        .SetColorAttachmentFormat(mSPSwapchain->GetFormat())
        .SetDepthStencilFormat(vk::Format::eUndefined)
        .Build("QuadDraw_Pipeline");

    DBG_LOG_INFO("Vulkan Quad Graphics Pipeline Created");
}

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};

    float flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));

    clearValue = {flash, flash, flash, 1.0f};

    auto subresource =
        Utils::GetDefaultImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

    cmd.clearColorImage(mDrawImage->GetHandle(), vk::ImageLayout::eGeneral,
                        clearValue, subresource);

    // Compute Draw
    {
        cmd.bindPipeline(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetComputePipeline("BackgroundCompute_Pipeline"));

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            mPipelineManager->GetLayoutHandle("BackgoundCompute_Layout"), 0,
            mDescriptorManager->GetDescriptor("DrawImage_Desc_0"), {});

        cmd.dispatch(::std::ceil(mDrawImage->GetExtent3D().width / 16.0),
                     ::std::ceil(mDrawImage->GetExtent3D().height / 16.0), 1);
    }

    // CUDA Draw
    // {
    //     auto layout = mDrawImage->GetLayout();
    //
    //     mDrawImage->TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    //     Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
    //                                  vk::ImageLayout::eUndefined,
    //                                  vk::ImageLayout::eTransferSrcOptimal);
    //
    //     vk::ImageBlit2 blitRegion {};
    //     blitRegion
    //         .setSrcOffsets(
    //             {vk::Offset3D {},
    //              vk::Offset3D {static_cast<int32_t>(
    //                                mCUDAExternalImage->GetExtent3D().width),
    //                            static_cast<int32_t>(
    //                                mCUDAExternalImage->GetExtent3D().height),
    //                            1}})
    //         .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
    //         .setDstOffsets(
    //             {vk::Offset3D {},
    //              vk::Offset3D {
    //                  static_cast<int32_t>(mDrawImage->GetExtent3D().width),
    //                  static_cast<int32_t>(mDrawImage->GetExtent3D().height),
    //                  1}})
    //         .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    //
    //     vk::BlitImageInfo2 blitInfo {};
    //     blitInfo.setDstImage(mDrawImage->GetHandle())
    //         .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
    //         .setSrcImage(mCUDAExternalImage->GetVkImage())
    //         .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
    //         .setFilter(vk::Filter::eLinear)
    //         .setRegions(blitRegion);
    //
    //     cmd.blitImage2(blitInfo);
    //
    //     mDrawImage->TransitionLayout(cmd, layout);
    //     Utils::TransitionImageLayout(cmd, mCUDAExternalImage->GetVkImage(),
    //                                  vk::ImageLayout::eTransferSrcOptimal,
    //                                  vk::ImageLayout::eGeneral);
    // }
}

void VulkanEngine::DrawMesh(vk::CommandBuffer cmd) {
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

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("TriangleDraw_Pipeline"));

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

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetLayoutHandle("Triangle_Layout"), 0,
        mDescriptorManager->GetDescriptor("Triangle_Desc_0"), {});

    MeshPushConstants pushConstants {};
    pushConstants.mVertexBufferAddress = mBoxMesh.mVertexBufferAddress;
    // mTriangleExternalMesh.mVertexBufferAddress;
    pushConstants.mModelMatrix = glm::mat4(1.0f);
    cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
                      vk::ShaderStageFlagBits::eVertex, 0,
                      sizeof(MeshPushConstants), &pushConstants);

    cmd.bindIndexBuffer(mBoxMesh.mIndexBuffer->GetHandle(), 0,
                        vk::IndexType::eUint32);

    cmd.drawIndexed(3, 1, 0, 0, 0);

    cmd.endRendering();
}

void VulkanEngine::DrawQuad(vk::CommandBuffer cmd) {
    auto imageIndex = mSPSwapchain->GetCurrentImageIndex();
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mSPSwapchain->GetImageViewHandle(imageIndex))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {{0, 0},
                                   {mSPSwapchain->GetExtent2D().width,
                                    mSPSwapchain->GetExtent2D().height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment);

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        mPipelineManager->GetGraphicsPipeline("QuadDraw_Pipeline"));

    vk::Viewport viewport {};
    viewport.setWidth(mSPSwapchain->GetExtent2D().width)
        .setHeight(mSPSwapchain->GetExtent2D().height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent({mSPSwapchain->GetExtent2D().width,
                       mSPSwapchain->GetExtent2D().height});
    cmd.setScissor(0, scissor);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           mPipelineManager->GetLayoutHandle("Quad_Layout"), 0,
                           mDescriptorManager->GetDescriptor("Quad_Desc_0"),
                           {});

    cmd.drawIndexed(3, 1, 0, 0, 0);

    cmd.endRendering();
}
