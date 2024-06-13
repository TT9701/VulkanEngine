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
vk::Bool32 VKAPI_PTR debugMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                            VkDebugUtilsMessageTypeFlagsEXT             messageTypes,
                                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                            void*                                       pUserData) {
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n", pCallbackData->pMessageIdName, pCallbackData->pMessage);
#if defined(_WIN32)
        __debugbreak();
#else
        raise(SIGTRAP);
#endif
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n", pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else {
        printf("MessageCode is %s & Message is %s \n", pCallbackData->pMessageIdName, pCallbackData->pMessage);
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
    vmaDestroyAllocator(mVmaAllocator);
    mDevice.destroy();
    mInstance.destroy(mSurface);
    mInstance.destroy(mDebugUtilsMessenger);
    mInstance.destroy();
}

void VulkanEngine::Draw() {}

void VulkanEngine::InitSDLWindow() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    mWindow = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, mWindowWidth,
                               mWindowHeight, window_flags);
}

void VulkanEngine::SetInstanceLayers(::std::vector<::std::string> const& requestedLayers) {
    auto                         instanceLayersProps = vk::enumerateInstanceLayerProperties();
    ::std::vector<::std::string> availableInstanceLayers {};
    for (auto& prop : instanceLayersProps) {
        availableInstanceLayers.push_back(prop.layerName);
    }
    auto available = Utils::FilterStringList(availableInstanceLayers, requestedLayers);
    mEnabledInstanceLayers.resize(available.size());
    ::std::transform(available.begin(), available.end(), mEnabledInstanceLayers.begin(),
                     ::std::mem_fn(&::std::string::c_str));
}

void VulkanEngine::SetInstanceExtensions(std::vector<std::string> const& requestedExtensions) {
    auto                         instanceExtensionProps = vk::enumerateInstanceExtensionProperties();
    ::std::vector<::std::string> availableInstanceExtensions {};
    for (auto& prop : instanceExtensionProps) {
        availableInstanceExtensions.push_back(prop.extensionName);
    }
    auto available = Utils::FilterStringList(availableInstanceExtensions, requestedExtensions);
    mEnabledInstanceExtensions.resize(available.size());
    ::std::transform(available.begin(), available.end(), mEnabledInstanceExtensions.begin(),
                     ::std::mem_fn(&::std::string::c_str));
}

std::vector<std::string> VulkanEngine::GetSDLRequestedInstanceExtensions() const {
    uint32_t count {0};
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count, nullptr);
    ::std::vector<const char*> requestedExtensions(count);
    SDL_Vulkan_GetInstanceExtensions(mWindow, &count, requestedExtensions.data());

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

    auto                         sdlRequestedInstanceExtensions = GetSDLRequestedInstanceExtensions();
    ::std::vector<::std::string> requestedInstanceExtensions {};
    requestedInstanceExtensions.insert(requestedInstanceExtensions.end(), sdlRequestedInstanceExtensions.begin(),
                                       sdlRequestedInstanceExtensions.end());
#ifdef DEBUG
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    SetInstanceExtensions(requestedInstanceExtensions);
    ::std::vector<const char*> enabledExtensionsCStr(mEnabledInstanceExtensions.size());
    for (int i = 0; i < mEnabledInstanceExtensions.size(); ++i) {
        enabledExtensionsCStr[i] = mEnabledInstanceExtensions[i].c_str();
    }

    vk::ApplicationInfo    appInfo {"Vulkan Engine", 1u, "Fun", 1u, VK_API_VERSION_1_3};
    vk::InstanceCreateInfo instanceCreateInfo {{},
                                               &appInfo,
                                               static_cast<uint32_t>(enabledLayersCStr.size()),
                                               enabledLayersCStr.data(),
                                               static_cast<uint32_t>(enabledExtensionsCStr.size()),
                                               enabledExtensionsCStr.data()};
    VK_CHECK(vk::createInstance(&instanceCreateInfo, nullptr, &mInstance));

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance);
#endif
}

#ifdef DEBUG
void VulkanEngine::CreateDebugUtilsMessenger() {
    const vk::DebugUtilsMessengerCreateInfoEXT messengerInfo {
        {},

        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
#if defined(VK_EXT_device_address_binding_report)
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding
#endif
        ,
        &debugMessengerCallback};
    mDebugUtilsMessenger = mInstance.createDebugUtilsMessengerEXT(messengerInfo);
}
#endif

void VulkanEngine::CreateSurface() {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    SDL_Vulkan_CreateSurface(mWindow, mInstance, &mSurface);
#endif
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
}

void VulkanEngine::SetQueueFamily(vk::QueueFlags requestedQueueTypes) {
    auto queueFamilyProps = mPhysicalDevice.getQueueFamilyProperties();
    for (uint32_t queueFamilyIndex = 0;
         queueFamilyIndex < queueFamilyProps.size() && (uint32_t)requestedQueueTypes != 0; ++queueFamilyIndex) {
        if (!mGraphicsFamilyIndex.has_value() &&
            (requestedQueueTypes & queueFamilyProps[queueFamilyIndex].queueFlags) & vk::QueueFlagBits::eGraphics) {
            mGraphicsFamilyIndex = queueFamilyIndex;
            mGraphicsQueueCount  = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eGraphics;
            continue;
        }

        if (!mComputeFamilyIndex.has_value() &&
            (requestedQueueTypes & queueFamilyProps[queueFamilyIndex].queueFlags) & vk::QueueFlagBits::eCompute) {
            mComputeFamilyIndex = queueFamilyIndex;
            mComputeQueueCount  = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eCompute;
            continue;
        }

        if (!mTransferFamilyIndex.has_value() &&
            (requestedQueueTypes & queueFamilyProps[queueFamilyIndex].queueFlags) & vk::QueueFlagBits::eTransfer) {
            mTransferFamilyIndex = queueFamilyIndex;
            mTransferQueueCount  = queueFamilyProps[queueFamilyIndex].queueCount;
            requestedQueueTypes &= ~vk::QueueFlagBits::eTransfer;
            continue;
        }
    }
}

void VulkanEngine::CreateDevice() {
    ::std::vector<float> queuePriorities(16, 1.0f);

    /**
     * TODO: Device layers & extensions
     */

    vk::PhysicalDeviceFeatures origFeatures {};

    vk::PhysicalDeviceVulkan13Features vulkan13Features {};
    vulkan13Features.setDynamicRendering(vk::True).setSynchronization2(vk::True);

    vk::PhysicalDeviceVulkan12Features vulkan12Features {};
    vulkan12Features.setBufferDeviceAddress(vk::True).setDescriptorIndexing(vk::True).setPNext(&vulkan13Features);

    vk::PhysicalDeviceVulkan11Features vulkan11Features {};
    vulkan11Features.setPNext(&vulkan12Features);

    ::std::vector<vk::DeviceQueueCreateInfo> queueCIs {};
    if (mGraphicsFamilyIndex.has_value())
        queueCIs.push_back({{}, mGraphicsFamilyIndex.value(), mGraphicsQueueCount, queuePriorities.data()});
    if (mComputeFamilyIndex.has_value())
        queueCIs.push_back({{}, mComputeFamilyIndex.value(), mComputeQueueCount, queuePriorities.data()});
    if (mTransferFamilyIndex.has_value())
        queueCIs.push_back({{}, mTransferFamilyIndex.value(), mTransferQueueCount, queuePriorities.data()});

    vk::DeviceCreateInfo deviceCreateInfo {
        {}, (uint32_t)queueCIs.size(), queueCIs.data(), {}, {}, {}, {}, &origFeatures, &vulkan11Features};
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
}

void VulkanEngine::CreateVmaAllocator() {
    const VmaVulkanFunctions vulkanFunctions = {
        .vkGetInstanceProcAddr               = vk::defaultDispatchLoaderDynamic.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr                 = vk::defaultDispatchLoaderDynamic.vkGetDeviceProcAddr,
        .vkGetPhysicalDeviceProperties       = vk::defaultDispatchLoaderDynamic.vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties = vk::defaultDispatchLoaderDynamic.vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory                    = vk::defaultDispatchLoaderDynamic.vkAllocateMemory,
        .vkFreeMemory                        = vk::defaultDispatchLoaderDynamic.vkFreeMemory,
        .vkMapMemory                         = vk::defaultDispatchLoaderDynamic.vkMapMemory,
        .vkUnmapMemory                       = vk::defaultDispatchLoaderDynamic.vkUnmapMemory,
        .vkFlushMappedMemoryRanges           = vk::defaultDispatchLoaderDynamic.vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges      = vk::defaultDispatchLoaderDynamic.vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory                  = vk::defaultDispatchLoaderDynamic.vkBindBufferMemory,
        .vkBindImageMemory                   = vk::defaultDispatchLoaderDynamic.vkBindImageMemory,
        .vkGetBufferMemoryRequirements       = vk::defaultDispatchLoaderDynamic.vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements        = vk::defaultDispatchLoaderDynamic.vkGetImageMemoryRequirements,
        .vkCreateBuffer                      = vk::defaultDispatchLoaderDynamic.vkCreateBuffer,
        .vkDestroyBuffer                     = vk::defaultDispatchLoaderDynamic.vkDestroyBuffer,
        .vkCreateImage                       = vk::defaultDispatchLoaderDynamic.vkCreateImage,
        .vkDestroyImage                      = vk::defaultDispatchLoaderDynamic.vkDestroyImage,
        .vkCmdCopyBuffer                     = vk::defaultDispatchLoaderDynamic.vkCmdCopyBuffer,
#if VMA_VULKAN_VERSION >= 1001000
        .vkGetBufferMemoryRequirements2KHR       = vk::defaultDispatchLoaderDynamic.vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR        = vk::defaultDispatchLoaderDynamic.vkGetImageMemoryRequirements2,
        .vkBindBufferMemory2KHR                  = vk::defaultDispatchLoaderDynamic.vkBindBufferMemory2,
        .vkBindImageMemory2KHR                   = vk::defaultDispatchLoaderDynamic.vkBindImageMemory2,
        .vkGetPhysicalDeviceMemoryProperties2KHR = vk::defaultDispatchLoaderDynamic.vkGetPhysicalDeviceMemoryProperties2,
#endif
#if VMA_VULKAN_VERSION >= 1003000
        .vkGetDeviceBufferMemoryRequirements = vk::defaultDispatchLoaderDynamic.vkGetDeviceBufferMemoryRequirements,
        .vkGetDeviceImageMemoryRequirements  = vk::defaultDispatchLoaderDynamic.vkGetDeviceImageMemoryRequirements,
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
}

void VulkanEngine::CreateSwapchain() {}
