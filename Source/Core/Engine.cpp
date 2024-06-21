#include "Engine.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include "Core/Utilities/VulkanUtilities.hpp"
#include "VulkanHelper.hpp"
#include "VulkanPipeline.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace {
#if defined(VK_EXT_debug_utils)
vk::Bool32 VKAPI_PTR debugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT             messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*                                       pUserData) {
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
#if defined(_WIN32)
        __debugbreak();
#else
        raise(SIGTRAP);
#endif
    } else if (messageSeverity &
               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    }

    return vk::False;
}
#endif
}  // namespace

void VulkanEngine::Init() {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif

    InitSDLWindow();
    InitVulkan();
}

void VulkanEngine::Run() {
    SDL_Event sdlEvent;

    bool bQuit = false;

    while (!bQuit) {
        while (SDL_PollEvent(&sdlEvent) != 0) {
            if (sdlEvent.type == SDL_QUIT)
                bQuit = true;
            if (sdlEvent.type == SDL_WINDOWEVENT) {
                if (sdlEvent.window.event == SDL_WINDOWEVENT_MINIMIZED)
                    mStopRendering = true;
                if (sdlEvent.window.event == SDL_WINDOWEVENT_RESTORED)
                    mStopRendering = false;
            }
            if (sdlEvent.type == SDL_KEYDOWN) {
                switch (sdlEvent.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        bQuit = true;
                    default:
                        break;
                }
            }
        }
        if (mStopRendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            Draw();
        }
    }
}

void VulkanEngine::Cleanup() {
    mDevice.waitIdle();

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mDevice.destroy(mFrameDatas[i].mCommandPool);
        mDevice.destroy(mFrameDatas[i].mRenderFence);
        mDevice.destroy(mFrameDatas[i].mReady4PresentSemaphore);
        mDevice.destroy(mFrameDatas[i].mReady4RenderSemaphore);

        mFrameDatas[i].mDeletionQueue.flush();
    }

    mMainDeletionQueue.flush();

    for (auto& view : mSwapchainImageViews)
        mDevice.destroy(view);
    mDevice.destroy(mSwapchain);

    mDevice.destroy();
    mInstance.destroy(mSurface);
    mInstance.destroy(mDebugUtilsMessenger);
    mInstance.destroy();
}

void VulkanEngine::Draw() {
    VK_CHECK(mDevice.waitForFences(GetCurrentFrameData().mRenderFence, vk::True,
                                   TIME_OUT_NANO_SECONDS));

    GetCurrentFrameData().mDeletionQueue.flush();
    mDevice.resetFences(GetCurrentFrameData().mRenderFence);

    uint32_t swapchainImageIndex;
    VK_CHECK(mDevice.acquireNextImageKHR(
        mSwapchain, TIME_OUT_NANO_SECONDS,
        GetCurrentFrameData().mReady4RenderSemaphore, VK_NULL_HANDLE,
        &swapchainImageIndex));

    auto cmd = GetCurrentFrameData().mCommandBuffer;

    cmd.reset();

    vk::CommandBufferBeginInfo cmdBeginInfo {
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    cmd.begin(cmdBeginInfo);

    mDrawImage.TransitionLayout(cmd, vk::ImageLayout::eGeneral);

    DrawBackground(cmd);

    mDrawImage.TransitionLayout(cmd, vk::ImageLayout::eColorAttachmentOptimal);

    DrawTriangle(cmd);

    mDrawImage.TransitionLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);

    Utils::TransitionImageLayout(cmd, mSwapchainImages[swapchainImageIndex],
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferDstOptimal);

    mDrawImage.CopyToImage(
        cmd, mSwapchainImages[swapchainImageIndex],
        {mDrawImage.mExtent3D.width, mDrawImage.mExtent3D.height},
        mSwapchainExtent);

    Utils::TransitionImageLayout(cmd, mSwapchainImages[swapchainImageIndex],
                                 vk::ImageLayout::eTransferDstOptimal,
                                 vk::ImageLayout::ePresentSrcKHR);

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

    mGraphicQueues[0].submit2(submit, GetCurrentFrameData().mRenderFence);

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(GetCurrentFrameData().mReady4PresentSemaphore)
        .setImageIndices(swapchainImageIndex);

    VK_CHECK(mGraphicQueues[0].presentKHR(presentInfo));

    cudaExternalSemaphoreWaitParams waitParams {};
    mCUDAStream.WaitExternalSemaphoresAsync(
        &mCUDAWaitSemaphore.GetCUDAExternalSemaphore(), &waitParams, 1);

    CUDA::SimPoint(mTriangleExternalMesh.mVertexBuffer
                       .GetMappedPointer(0, 3 * sizeof(Vertex))
                       .GetPtr(),
                   mFrameNum, mCUDAStream.Get());

    CUDA::SimSurface(*mCUDAExternalImage.GetSurfaceObjectPtr(), mFrameNum,
                     mCUDAStream.Get());

    cudaExternalSemaphoreSignalParams signalParams {};
    mCUDAStream.SignalExternalSemaphoresAsyn(
        &mCUDASignalSemaphore.GetCUDAExternalSemaphore(), &signalParams, 1);

    ++mFrameNum;
}

void VulkanEngine::InitSDLWindow() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    mWindow = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED, mWindowWidth,
                               mWindowHeight, window_flags);
    DBG_LOG_INFO("SDL_Window Created. Width: %d, Height: %d.", mWindowWidth,
                 mWindowHeight);
}

void VulkanEngine::SetInstanceLayers(
    ::std::vector<::std::string> const& requestedLayers) {
    auto instanceLayersProps = vk::enumerateInstanceLayerProperties();
    ::std::vector<::std::string> availableInstanceLayers {};
    for (auto& prop : instanceLayersProps) {
        availableInstanceLayers.push_back(prop.layerName);
    }
    auto available =
        Utils::FilterStringList(availableInstanceLayers, requestedLayers);
    mEnabledInstanceLayers.resize(available.size());
    std::ranges::transform(available, mEnabledInstanceLayers.begin(),
                           ::std::mem_fn(&::std::string::c_str));
}

void VulkanEngine::SetInstanceExtensions(
    std::vector<std::string> const& requestedExtensions) {
    auto instanceExtensionProps = vk::enumerateInstanceExtensionProperties();
    ::std::vector<::std::string> availableInstanceExtensions {};
    for (auto& prop : instanceExtensionProps) {
        availableInstanceExtensions.push_back(prop.extensionName);
    }
    auto available = Utils::FilterStringList(availableInstanceExtensions,
                                             requestedExtensions);
    mEnabledInstanceExtensions.resize(available.size());
    std::ranges::transform(available, mEnabledInstanceExtensions.begin(),
                           ::std::mem_fn(&::std::string::c_str));
}

std::vector<std::string> VulkanEngine::GetSDLRequestedInstanceExtensions()
    const {
    uint32_t count {0};
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count, nullptr);
    ::std::vector<const char*> requestedExtensions(count);
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count,
                                     requestedExtensions.data());

    std::vector<std::string> result {};
    result.reserve(requestedExtensions.size());
    for (auto& ext : requestedExtensions) {
        result.emplace_back(ext);
    }
    return result;
}

void VulkanEngine::InitVulkan() {
    CreateInstance();
    CreateDebugUtilsMessenger();
    CreateSurface();
    PickPhysicalDevice();
    CreateDevice();
    CreateVmaAllocators();
    CreateSwapchain();
    CreateCommands();
    CreateSyncStructures();

    CreateErrorCheckTextures();
    CreateDefaultSamplers();

    CreateDescriptors();
    CreatePipelines();

    CreateTriangleData();
    CreateExternalTriangleData();

    SetCudaInterop();
}

void VulkanEngine::CreateInstance() {
    ::std::vector<::std::string> requestedInstanceLayers {};
#ifdef DEBUG
    requestedInstanceLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif
    SetInstanceLayers(requestedInstanceLayers);
    ::std::vector<const char*> enabledLayersCStr(mEnabledInstanceLayers.size());
    for (int i = 0; i < mEnabledInstanceLayers.size(); ++i) {
        enabledLayersCStr[i] = mEnabledInstanceLayers[i].c_str();
    }

    auto sdlRequestedInstanceExtensions = GetSDLRequestedInstanceExtensions();
    ::std::vector<::std::string> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(),
                                       sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifdef DEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    SetInstanceExtensions(requestedInstanceExtensions);
    ::std::vector<const char*> enabledExtensionsCStr(
        mEnabledInstanceExtensions.size());
    for (int i = 0; i < mEnabledInstanceExtensions.size(); ++i) {
        enabledExtensionsCStr[i] = mEnabledInstanceExtensions[i].c_str();
    }

    vk::ApplicationInfo appInfo {};
    appInfo.setPEngineName("Vulkan Engine")
        .setPApplicationName("Fun")
        .setEngineVersion(1u)
        .setApplicationVersion(1u)
        .setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceCreateInfo {};
    instanceCreateInfo.setPApplicationInfo(&appInfo)
        .setPEnabledLayerNames(enabledLayersCStr)
        .setPEnabledExtensionNames(enabledExtensionsCStr);
    mInstance = vk::createInstance(instanceCreateInfo);

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance);
#endif

    DBG_LOG_INFO("Vulkan Instance Created");
}

#ifdef DEBUG
void VulkanEngine::CreateDebugUtilsMessenger() {
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo {};
    messengerInfo
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
        .setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
#if defined(VK_EXT_device_address_binding_report)
            vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
#endif
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(&debugMessengerCallback);
    mDebugUtilsMessenger =
        mInstance.createDebugUtilsMessengerEXT(messengerInfo);

    DBG_LOG_INFO("Vulkan Debug Messenger Created");
}
#endif

void VulkanEngine::CreateSurface() {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(mWindow, mInstance, &mSurface);
#endif
    DBG_LOG_INFO("SDL Vulkan Surface Created");
}

void VulkanEngine::PickPhysicalDevice() {
    auto deviceList = mInstance.enumeratePhysicalDevices();
    assert(!deviceList.empty(), "device list is empty");

    for (auto& device : deviceList) {
        std::string devicename(device.getProperties().deviceName.data());
        const auto  result = devicename.find("NVIDIA");
        if (result != std::string::npos) {
            mPhysicalDevice = device;
            break;
        }
    }
    SetQueueFamily(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);

    DBG_LOG_INFO("Physical Device Selected: %s",
                 mPhysicalDevice.getProperties().deviceName.data());
}

void VulkanEngine::SetQueueFamily(vk::QueueFlags requestedQueueTypes) {
    auto queueFamilyProps = mPhysicalDevice.getQueueFamilyProperties();
    for (uint32_t queueFamilyIndex = 0;
         queueFamilyIndex < queueFamilyProps.size() &&
         static_cast<uint32_t>(requestedQueueTypes) != 0;
         ++queueFamilyIndex) {
        if (!mGraphicsFamilyIndex.has_value() &&
            (requestedQueueTypes &
             queueFamilyProps[queueFamilyIndex].queueFlags) &
                vk::QueueFlagBits::eGraphics) {
            mGraphicsFamilyIndex = queueFamilyIndex;
            mGraphicsQueueCount = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eGraphics;
            continue;
        }

        if (!mComputeFamilyIndex.has_value() &&
            (requestedQueueTypes &
             queueFamilyProps[queueFamilyIndex].queueFlags) &
                vk::QueueFlagBits::eCompute) {
            mComputeFamilyIndex = queueFamilyIndex;
            mComputeQueueCount  = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eCompute;
            continue;
        }

        if (!mTransferFamilyIndex.has_value() &&
            (requestedQueueTypes &
             queueFamilyProps[queueFamilyIndex].queueFlags) &
                vk::QueueFlagBits::eTransfer) {
            mTransferFamilyIndex = queueFamilyIndex;
            mTransferQueueCount = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eTransfer;
        }
    }
}

void VulkanEngine::CreateDevice() {
    ::std::vector<float> queuePriorities(16, 1.0f);

    /**
     * TODO: Device layers & extensions
     */

    auto availableLayers = mPhysicalDevice.enumerateDeviceLayerProperties();
    auto availableExtensions =
        mPhysicalDevice.enumerateDeviceExtensionProperties();

    // for (auto& ext : availableExtensions) {
    //     ::std::cout << ext.extensionName << "\n";
    // }

    ::std::vector<const char*> enabledDeviceLayers {};
    ::std::vector<const char*> enabledDeivceExtensions {};

    enabledDeivceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabledDeivceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledDeivceExtensions.push_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledDeivceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    enabledDeivceExtensions.push_back(
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

    ::std::vector<vk::DeviceQueueCreateInfo> queueCIs {};
    if (mGraphicsFamilyIndex.has_value())
        queueCIs.push_back({{},
                            mGraphicsFamilyIndex.value(),
                            mGraphicsQueueCount,
                            queuePriorities.data()});
    if (mComputeFamilyIndex.has_value())
        queueCIs.push_back({{},
                            mComputeFamilyIndex.value(),
                            mComputeQueueCount,
                            queuePriorities.data()});
    if (mTransferFamilyIndex.has_value())
        queueCIs.push_back({{},
                            mTransferFamilyIndex.value(),
                            mTransferQueueCount,
                            queuePriorities.data()});

    vk::DeviceCreateInfo deviceCreateInfo {};
    deviceCreateInfo.setQueueCreateInfos(queueCIs)
        .setPEnabledLayerNames(enabledDeviceLayers)
        .setPEnabledExtensionNames(enabledDeivceExtensions)
        .setPEnabledFeatures(&origFeatures)
        .setPNext(&vulkan11Features);
    mDevice = mPhysicalDevice.createDevice(deviceCreateInfo);

    mGraphicQueues.resize(mGraphicsQueueCount);
    for (int i = 0; i < mGraphicsQueueCount; ++i)
        mGraphicQueues[i] = mDevice.getQueue(mGraphicsFamilyIndex.value(), i);

    mComputeQueues.resize(mComputeQueueCount);
    for (int i = 0; i < mComputeQueueCount; ++i)
        mComputeQueues[i] = mDevice.getQueue(mComputeFamilyIndex.value(), i);

    mTransferQueues.resize(mTransferQueueCount);
    for (int i = 0; i < mTransferQueueCount; ++i)
        mTransferQueues[i] = mDevice.getQueue(mTransferFamilyIndex.value(), i);

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice);
#endif
    DBG_LOG_INFO("Vulkan Device Created");
}

void VulkanEngine::CreateVmaAllocators() {
    const VmaVulkanFunctions vulkanFunctions = {
        .vkGetInstanceProcAddr =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
        .vkGetPhysicalDeviceProperties =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory,
        .vkFreeMemory     = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory,
        .vkMapMemory      = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory,
        .vkUnmapMemory    = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory,
        .vkFlushMappedMemoryRanges =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory,
        .vkBindImageMemory  = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory,
        .vkGetBufferMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements,
        .vkCreateBuffer  = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer,
        .vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer,
        .vkCreateImage   = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage,
        .vkDestroyImage  = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage,
        .vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer,
#if VMA_VULKAN_VERSION >= 1001000
        .vkGetBufferMemoryRequirements2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2,
        .vkBindBufferMemory2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2,
        .vkBindImageMemory2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2,
        .vkGetPhysicalDeviceMemoryProperties2KHR =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2,
#endif
#if VMA_VULKAN_VERSION >= 1003000
        .vkGetDeviceBufferMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceBufferMemoryRequirements,
        .vkGetDeviceImageMemoryRequirements =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements,
#endif
    };

    VmaAllocatorCreateInfo allocInfo = {
#if defined(VK_KHR_buffer_device_address) && defined(_WIN32)
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
#endif
        .physicalDevice   = mPhysicalDevice,
        .device           = mDevice,
        .pVulkanFunctions = &vulkanFunctions,
        .instance         = mInstance,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    vmaCreateAllocator(&allocInfo, &mVmaAllocator);

    mMainDeletionQueue.push_function(
        [&]() { vmaDestroyAllocator(mVmaAllocator); });

    DBG_LOG_INFO("vma Allocator Created");

    VmaPoolCreateInfo vmaPoolCreateInfo {};
    vmaPoolCreateInfo.pMemoryAllocateNext = &mExportMemoryAllocateInfo;

    vmaCreatePool(mVmaAllocator, &vmaPoolCreateInfo, &mVmaExternalMemoryPool);

    mMainDeletionQueue.push_function(
        [&]() { vmaDestroyPool(mVmaAllocator, mVmaExternalMemoryPool); });

    DBG_LOG_INFO("vma External Resource Pool Created");
}

void VulkanEngine::CreateSwapchain() {
    mSwapchainImageFormat = vk::Format::eR8G8B8A8Unorm;
    mSwapchainExtent      = vk::Extent2D {static_cast<uint32_t>(mWindowWidth),
                                     static_cast<uint32_t>(mWindowHeight)};

    vk::SwapchainCreateInfoKHR swapchainCreateInfo {};
    swapchainCreateInfo.setSurface(mSurface)
        .setMinImageCount(3u)
        .setImageFormat(mSwapchainImageFormat)
        .setImageExtent(mSwapchainExtent)
        .setImageArrayLayers(1u)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment |
                       vk::ImageUsageFlagBits::eTransferDst)
        .setPresentMode(vk::PresentModeKHR::eMailbox)
        .setClipped(vk::True)
        .setOldSwapchain(VK_NULL_HANDLE);
    mSwapchain = mDevice.createSwapchainKHR(swapchainCreateInfo);

    mSwapchainImages = mDevice.getSwapchainImagesKHR(mSwapchain);

    mSwapchainImageViews.resize(mSwapchainImages.size());
    for (int i = 0; i < mSwapchainImages.size(); ++i) {
        vk::ImageViewCreateInfo imgViewCreateInfo {};
        imgViewCreateInfo.setImage(mSwapchainImages[i])
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(mSwapchainImageFormat)
            .setSubresourceRange(vk::ImageSubresourceRange {
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        mSwapchainImageViews[i] = mDevice.createImageView(imgViewCreateInfo);
    }

    DBG_LOG_INFO(
        "Vulkan Swapchain Created. PresentMode: %s. \n\t\t\t    "
        "Swapchain Image Count: %d",
        vk::to_string(swapchainCreateInfo.presentMode).c_str(),
        mSwapchainImages.size());

    vk::Extent3D drawImageExtent {static_cast<uint32_t>(mWindowWidth),
                                  static_cast<uint32_t>(mWindowHeight), 1};

    vk::ImageUsageFlags drawImageUsage {};
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    drawImageUsage |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsage |= vk::ImageUsageFlagBits::eColorAttachment;

    mDrawImage.CreateImage(mDevice, mVmaAllocator,
                           VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                           drawImageExtent, vk::Format::eR16G16B16A16Sfloat,
                           drawImageUsage, vk::ImageAspectFlagBits::eColor);

    mMainDeletionQueue.push_function(
        [&]() { mDrawImage.Destroy(mDevice, mVmaAllocator); });

    mCUDAExternalImage.CreateExternalImage(
        mDevice, mVmaAllocator, mVmaExternalMemoryPool, 0, drawImageExtent,
        vk::Format::eR32G32B32A32Sfloat, drawImageUsage,
        vk::ImageAspectFlagBits::eColor);

    mMainDeletionQueue.push_function(
        [&]() { mCUDAExternalImage.Destroy(mDevice, mVmaAllocator); });
}

void VulkanEngine::CreateCommands() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(mGraphicsFamilyIndex.value());

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mFrameDatas[i].mCommandPool =
            mDevice.createCommandPool(cmdPoolCreateInfo);

        vk::CommandBufferAllocateInfo cmdAllocInfo {};
        cmdAllocInfo.setCommandPool(mFrameDatas[i].mCommandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1u);
        mFrameDatas[i].mCommandBuffer =
            mDevice.allocateCommandBuffers(cmdAllocInfo)[0];
    }

    DBG_LOG_INFO("Vulkan Per Frame CommandPool & CommandBuffer Created");

    mImmediateSubmit.mCommandPool =
        mDevice.createCommandPool(cmdPoolCreateInfo);

    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(mImmediateSubmit.mCommandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1u);
    mImmediateSubmit.mCommandBuffer =
        mDevice.allocateCommandBuffers(cmdAllocInfo)[0];

    mMainDeletionQueue.push_function(
        [=]() { mDevice.destroy(mImmediateSubmit.mCommandPool); });

    DBG_LOG_INFO("Vulkan Immediate submit CommandPool & CommandBuffer Created");
}

void VulkanEngine::CreateSyncStructures() {
    vk::FenceCreateInfo fenceCreateInfo {vk::FenceCreateFlagBits::eSignaled};
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        mFrameDatas[i].mRenderFence = mDevice.createFence(fenceCreateInfo);
        mFrameDatas[i].mReady4PresentSemaphore =
            mDevice.createSemaphore(semaphoreCreateInfo);
        mFrameDatas[i].mReady4RenderSemaphore =
            mDevice.createSemaphore(semaphoreCreateInfo);
    }

    DBG_LOG_INFO("Vulkan Per Frame Fence & Semaphore Created");

    mImmediateSubmit.mFence = mDevice.createFence(fenceCreateInfo);
    mMainDeletionQueue.push_function(
        [=]() { mDevice.destroy(mImmediateSubmit.mFence); });

    DBG_LOG_INFO("Vulkan Immediate submit Fence & Semaphore Created");

    mCUDAWaitSemaphore.CreateExternalSemaphore(mDevice);
    mCUDASignalSemaphore.CreateExternalSemaphore(mDevice);

    // mCUDAWaitSemaphore.InsertSignalToStreamAsync()

    mMainDeletionQueue.push_function([=]() {
        mCUDASignalSemaphore.Destroy(mDevice);
        mCUDAWaitSemaphore.Destroy(mDevice);
    });

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

    mMainDescriptorAllocator.InitPool(mDevice, 10, sizes);

    CreateBackgroundComputeDescriptors();
    CreateTriangleDescriptors();

    mMainDeletionQueue.push_function(
        [&]() { mMainDescriptorAllocator.DestroyPool(mDevice); });
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

    mMainDeletionQueue.push_function([&]() {
        mTriangleMesh.mVertexBuffer.Destroy();
        mTriangleMesh.mIndexBuffer.Destroy();
    });
}

void VulkanEngine::CreateExternalTriangleData() {
    CUDA::VulkanExternalBuffer vertexBuffer {}, indexBuffer {};
    vertexBuffer.CreateExternalBuffer(
        mDevice, mVmaAllocator, 3 * sizeof(Vertex),
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, mVmaExternalMemoryPool);

    indexBuffer.CreateExternalBuffer(
        mDevice, mVmaAllocator, 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, mVmaExternalMemoryPool);

    mTriangleExternalMesh.mVertexBuffer = vertexBuffer;
    mTriangleExternalMesh.mIndexBuffer  = indexBuffer;

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(mTriangleExternalMesh.mVertexBuffer.GetVkBuffer());

    mTriangleExternalMesh.mVertexBufferAddress =
        mDevice.getBufferAddress(deviceAddrInfo);

    mMainDeletionQueue.push_function([&]() {
        mTriangleExternalMesh.mVertexBuffer.Destroy();
        mTriangleExternalMesh.mIndexBuffer.Destroy();
    });

    CUDA::VulkanExternalImage externalImage {};
    externalImage.CreateExternalImage(
        mDevice, mVmaAllocator, mVmaExternalMemoryPool, 0, {16, 16, 1},
        vk::Format::eR8G8B8A8Uint, vk::ImageUsageFlagBits::eStorage,
        vk::ImageAspectFlagBits::eColor);

    auto cudaMipmapped = externalImage.GetMapMipmappedArray(0, 1);

    externalImage.Destroy(mDevice, mVmaAllocator);
}

void VulkanEngine::CreateErrorCheckTextures() {
    uint32_t black   = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;  //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    mErrorCheckImage.CreateImage(
        pixels.data(), this, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VkExtent3D {16, 16, 1}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor);

    mMainDeletionQueue.push_function(
        [&]() { mErrorCheckImage.Destroy(mDevice, mVmaAllocator); });
}

void VulkanEngine::CreateDefaultSamplers() {
    vk::SamplerCreateInfo info {};
    info.setMinFilter(vk::Filter::eNearest).setMagFilter(vk::Filter::eNearest);
    mDefaultSamplerNearest = mDevice.createSampler(info);
    info.setMinFilter(vk::Filter::eLinear).setMagFilter(vk::Filter::eLinear);
    mDefaultSamplerLinear = mDevice.createSampler(info);

    mMainDeletionQueue.push_function([&]() {
        mDevice.destroy(mDefaultSamplerLinear);
        mDevice.destroy(mDefaultSamplerNearest);
    });
}

void VulkanEngine::SetCudaInterop() {
    auto result = CUDA::GetVulkanCUDABindDeviceID(mPhysicalDevice);
    DBG_LOG_INFO("Cuda Interop: physical device uuid: %d", result);
}

GPUMeshBuffers VulkanEngine::UploadMeshData(std::span<uint32_t> indices,
                                            std::span<Vertex>   vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize  = indices.size() * sizeof(uint32_t);

    AllocatedVulkanBuffer vertexBuffer {}, indexBuffer {};
    vertexBuffer.CreateBuffer(mVmaAllocator, vertexBufferSize,
                              vk::BufferUsageFlagBits::eStorageBuffer |
                                  vk::BufferUsageFlagBits::eTransferDst |
                                  vk::BufferUsageFlagBits::eShaderDeviceAddress,
                              VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    indexBuffer.CreateBuffer(mVmaAllocator, indexBufferSize,
                             vk::BufferUsageFlagBits::eIndexBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst,
                             VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    GPUMeshBuffers newMesh {};
    newMesh.mVertexBuffer = vertexBuffer;
    newMesh.mIndexBuffer  = indexBuffer;

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(newMesh.mVertexBuffer.mBuffer);

    newMesh.mVertexBufferAddress = mDevice.getBufferAddress(deviceAddrInfo);

    AllocatedVulkanBuffer staging {};
    staging.CreateBuffer(
        mVmaAllocator, vertexBufferSize + indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    void* data = staging.mInfo.pMappedData;
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    ImmediateSubmit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging.mBuffer, newMesh.mVertexBuffer.mBuffer,
                       vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging.mBuffer, newMesh.mIndexBuffer.mBuffer,
                       indexCopy);
    });

    staging.Destroy();

    return newMesh;
}

void VulkanEngine::ImmediateSubmit(
    std::function<void(vk::CommandBuffer cmd)>&& function) {
    mDevice.resetFences(mImmediateSubmit.mFence);

    auto cmd = mImmediateSubmit.mCommandBuffer;

    cmd.reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);

    function(cmd);

    cmd.end();

    auto cmdSubmitInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);
    auto submit        = Utils::SubmitInfo(cmdSubmitInfo, {}, {});

    mGraphicQueues[0].submit2(submit, mImmediateSubmit.mFence);
    VK_CHECK(mDevice.waitForFences(mImmediateSubmit.mFence, vk::True,
                                   TIME_OUT_NANO_SECONDS));
}

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    DescriptorLayoutBuilder builder;
    builder.AddBinding(0, vk::DescriptorType::eStorageImage);
    mDrawImageDescriptorLayout =
        builder.Build(mDevice, vk::ShaderStageFlagBits::eCompute);

    mDrawImageDescriptors =
        mMainDescriptorAllocator.Allocate(mDevice, mDrawImageDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(
        0, {VK_NULL_HANDLE, mDrawImage.mImageView, vk::ImageLayout::eGeneral},
        vk::DescriptorType::eStorageImage);

    writer.UpdateSet(mDevice, mDrawImageDescriptors);

    mMainDeletionQueue.push_function(
        [&]() { mDevice.destroy(mDrawImageDescriptorLayout); });

    DBG_LOG_INFO("Vulkan Background Compute Descriptors Created");
}

void VulkanEngine::CreateBackgroundComputePipeline() {
    vk::PipelineLayoutCreateInfo computeLayout {};
    computeLayout.setSetLayouts(mDrawImageDescriptorLayout);

    mBackgroundComputePipelineLayout =
        mDevice.createPipelineLayout(computeLayout);

    vk::ShaderModule computeDrawShader {};
    assert(Utils::LoadShaderModule("../../Shaders/BackGround.comp.spv", mDevice,
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
        mDevice.createComputePipeline({}, computePipelineCreateInfo).value;

    mDevice.destroy(computeDrawShader);

    mMainDeletionQueue.push_function([&]() {
        mDevice.destroy(mBackgroundComputePipelineLayout);
        mDevice.destroy(mBackgroundComputePipeline);
    });

    DBG_LOG_INFO("Vulkan Background Compute Pipeline Created");
}

void VulkanEngine::CreateTrianglePipeline() {
    vk::ShaderModule vertexShader {};
    vk::ShaderModule fragmentShader {};

    assert(Utils::LoadShaderModule("../../Shaders/Triangle.vert.spv", mDevice,
                                   &vertexShader),
           "Error when building the triangle vertex shader");
    assert(Utils::LoadShaderModule("../../Shaders/Triangle.frag.spv", mDevice,
                                   &fragmentShader),
           "Error when building the triangle fragment shader");

    vk::PushConstantRange pushConstant {};
    pushConstant.setSize(sizeof(MeshPushConstants))
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    vk::PipelineLayoutCreateInfo layoutCreateInfo {};
    layoutCreateInfo.setPushConstantRanges(pushConstant)
        .setSetLayouts(mTextureTriangleDescriptorLayout);
    mTrianglePipelieLayout = mDevice.createPipelineLayout(layoutCreateInfo);

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
            .SetColorAttachmentFormat(mDrawImage.mFormat)
            .SetDepthStencilFormat(vk::Format::eUndefined)
            .Build(mDevice);

    mDevice.destroy(vertexShader);
    mDevice.destroy(fragmentShader);

    mMainDeletionQueue.push_function([&]() {
        mDevice.destroy(mTrianglePipelieLayout);
        mDevice.destroy(mTrianglePipelie);
    });
    DBG_LOG_INFO("Vulkan Triagnle Graphics Pipeline Created");
}

void VulkanEngine::CreateTriangleDescriptors() {
    DescriptorLayoutBuilder builder {};
    builder.AddBinding(0, vk::DescriptorType::eCombinedImageSampler);
    mTextureTriangleDescriptorLayout =
        builder.Build(mDevice, vk::ShaderStageFlagBits::eFragment);

    mTextureTriangleDescriptors = mMainDescriptorAllocator.Allocate(
        mDevice, mTextureTriangleDescriptorLayout);

    DescriptorWriter writer {};
    writer.WriteImage(0,
                      {mDefaultSamplerNearest, mErrorCheckImage.mImageView,
                       vk::ImageLayout::eShaderReadOnlyOptimal},
                      vk::DescriptorType::eCombinedImageSampler);

    writer.UpdateSet(mDevice, mTextureTriangleDescriptors);

    mMainDeletionQueue.push_function(
        [&]() { mDevice.destroy(mTextureTriangleDescriptorLayout); });
}

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};
    float               flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));
    clearValue                = {flash, flash, flash, 1.0f};

    auto subresource =
        Utils::GetDefaultImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

    cmd.clearColorImage(mDrawImage.mImage, vk::ImageLayout::eGeneral,
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

    auto layout = mDrawImage.mLayout;

    mDrawImage.TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    Utils::TransitionImageLayout(cmd, mCUDAExternalImage.GetVkImage(),
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferSrcOptimal);

    vk::ImageBlit2 blitRegion {};
    blitRegion
        .setSrcOffsets(
            {vk::Offset3D {},
             vk::Offset3D {
                 static_cast<int32_t>(mCUDAExternalImage.GetExtent3D().width),
                 static_cast<int32_t>(mCUDAExternalImage.GetExtent3D().height),
                 1}})
        .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setDstOffsets(
            {vk::Offset3D {},
             vk::Offset3D {static_cast<int32_t>(mDrawImage.mExtent3D.width),
                           static_cast<int32_t>(mDrawImage.mExtent3D.height),
                           1}})
        .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});

    vk::BlitImageInfo2 blitInfo {};
    blitInfo.setDstImage(mDrawImage.mImage)
        .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcImage(mCUDAExternalImage.GetVkImage())
        .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setFilter(vk::Filter::eLinear)
        .setRegions(blitRegion);

    cmd.blitImage2(blitInfo);

    mDrawImage.TransitionLayout(cmd, layout);
    Utils::TransitionImageLayout(cmd, mCUDAExternalImage.GetVkImage(),
                                 vk::ImageLayout::eTransferSrcOptimal,
                                 vk::ImageLayout::eGeneral);
}

void VulkanEngine::DrawTriangle(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment {};
    colorAttachment.setImageView(mDrawImage.mImageView)
        .setImageLayout(mDrawImage.mLayout)
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderInfo {};
    renderInfo
        .setRenderArea(vk::Rect2D {
            {0, 0}, {mDrawImage.mExtent3D.width, mDrawImage.mExtent3D.height}})
        .setLayerCount(1u)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(VK_NULL_HANDLE)     // TODO
        .setPStencilAttachment(VK_NULL_HANDLE);  // TODO

    cmd.beginRendering(renderInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mTrianglePipelie);

    vk::Viewport viewport {};
    viewport.setWidth(mDrawImage.mExtent3D.width)
        .setHeight(mDrawImage.mExtent3D.height)
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor {};
    scissor.setExtent(
        {mDrawImage.mExtent3D.width, mDrawImage.mExtent3D.height});
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

    cmd.bindIndexBuffer(mTriangleMesh.mIndexBuffer.mBuffer, 0,
                        vk::IndexType::eUint32);

    cmd.drawIndexed(3, 1, 0, 0, 0);

    cmd.endRendering();
}
