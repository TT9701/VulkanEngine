#include "Engine.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include "Core/Utilities/VulkanUtilities.hpp"
#include "VulkanHelper.hpp"

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

    Utils::TransitionImageLayout(cmd, mDrawImage.mImage,
                                 vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eGeneral);

    DrawBackground(cmd);

    Utils::TransitionImageLayout(cmd, mDrawImage.mImage,
                                 vk::ImageLayout::eGeneral,
                                 vk::ImageLayout::eTransferSrcOptimal);

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

    auto cmdInfo  = Utils::GetDefaultCommandBufferSubmitInfo(cmd);
    auto waitInfo = Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        GetCurrentFrameData().mReady4RenderSemaphore);
    auto signalInfo = Utils::GetDefaultSemaphoreSubmitInfo(
        vk::PipelineStageFlagBits2::eAllGraphics,
        GetCurrentFrameData().mReady4PresentSemaphore);
    auto submit = Utils::SubmitInfo(cmdInfo, signalInfo, waitInfo);

    mGraphicQueues[0].submit2(submit, GetCurrentFrameData().mRenderFence);

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(GetCurrentFrameData().mReady4PresentSemaphore)
        .setImageIndices(swapchainImageIndex);

    VK_CHECK(mGraphicQueues[0].presentKHR(presentInfo));

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
    CreateVmaAllocator();
    CreateSwapchain();
    CreateCommands();
    CreateSyncStructures();
    CreateBackgroundComputeDescriptors();
    CreatePipelines();
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

    ::std::vector<const char*> enabledDeviceLayers {};
    ::std::vector<const char*> enabledDeivceExtensions {};

    enabledDeivceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

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

void VulkanEngine::CreateVmaAllocator() {
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

    const VmaAllocatorCreateInfo allocInfo = {
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

    /*https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html*/
    VmaAllocationCreateInfo imageAllocInfo {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    imageAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    mDrawImage.CreateImage(mVmaAllocator, imageAllocInfo, drawImageExtent,
                           vk::Format::eR16G16B16A16Sfloat, drawImageUsage);

    mDrawImage.CreateImageView(mDevice, vk::ImageAspectFlagBits::eColor);

    mMainDeletionQueue.push_function(
        [&]() { mDrawImage.Destroy(mDevice, mVmaAllocator); });
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
}

void VulkanEngine::CreatePipelines() {
    CreateBackgroundComputePipeline();
}

void VulkanEngine::CreateBackgroundComputeDescriptors() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes {
        {vk::DescriptorType::eStorageImage, 1}};

    mMainDescriptorAllocator.InitPool(mDevice, 10, sizes);

    DescriptorLayoutBuilder builder;
    builder.AddBinding(0, vk::DescriptorType::eStorageImage);
    mDrawImageDescriptorLayout =
        builder.Build(mDevice, vk::ShaderStageFlagBits::eCompute);

    mDrawImageDescriptors =
        mMainDescriptorAllocator.Allocate(mDevice, mDrawImageDescriptorLayout);

    vk::DescriptorImageInfo imgInfo {};
    imgInfo.setImageLayout(vk::ImageLayout::eGeneral)
        .setImageView(mDrawImage.mImageView);

    vk::WriteDescriptorSet drawImageWrite {};
    drawImageWrite.setDstBinding(0)
        .setDstSet(mDrawImageDescriptors.front())
        .setDescriptorCount(1u)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imgInfo);

    mDevice.updateDescriptorSets(drawImageWrite, {});

    mMainDeletionQueue.push_function([&]() {
        mMainDescriptorAllocator.DestroyPool(mDevice);
        mDevice.destroy(mDrawImageDescriptorLayout);
    });
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
}

void VulkanEngine::DrawBackground(vk::CommandBuffer cmd) {
    vk::ClearColorValue clearValue {};
    float               flash = ::std::fabs(::std::sin(mFrameNum / 6000.0f));
    clearValue                = {flash, flash, flash, 1.0f};

    auto subresource =
        Utils::GetDefaultImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

    cmd.clearColorImage(mDrawImage.mImage, vk::ImageLayout::eGeneral,
                        clearValue, subresource);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute,
                     mBackgroundComputePipeline);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                           mBackgroundComputePipelineLayout, 0,
                           mDrawImageDescriptors, {});

    cmd.dispatch(::std::ceil(mDrawImage.mExtent3D.width / 16.0),
                 ::std::ceil(mDrawImage.mExtent3D.height / 16.0), 1);
}
