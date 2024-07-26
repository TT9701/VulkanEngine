#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanHelper.hpp"
#ifndef NDEBUG
#include "VulkanDebugUtils.hpp"
#endif
#include "VulkanInstance.hpp"
#include "VulkanMemoryAllocator.hpp"
#include "VulkanPhysicalDevice.hpp"
#include "VulkanSurface.hpp"

class SDLWindow;

class VulkanContext {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanContext(
        const Type_SPInstance<SDLWindow>& window,
        vk::QueueFlags requestedQueueFlags,
        ::std::vector<::std::string> const& requestedInstanceLayers = {},
        ::std::vector<::std::string> const& requestedInstanceExtensions = {},
        ::std::vector<::std::string> const& requestedDeviceExtensions = {});
    ~VulkanContext() = default;
    MOVABLE_ONLY(VulkanContext);

public:
    Type_SPInstance<VulkanInstance> GetInstance() const { return mSPInstance; }

#ifndef NDEBUG
    Type_SPInstance<VulkanDebugUtils> GetDebugMessenger() const {
        return mSPDebugUtilsMessenger;
    }
#endif

    Type_SPInstance<VulkanSurface> GetSurface() const { return mSPSurface; }

    Type_SPInstance<VulkanPhysicalDevice> GetPhysicalDevice() const {
        return mSPPhysicalDevice;
    }

    Type_SPInstance<VulkanDevice> GetDevice() const { return mSPDevice; }

    Type_SPInstance<VulkanMemoryAllocator> GetVmaAllocator() const {
        return mSPAllocator;
    }

#ifdef CUDA_VULKAN_INTEROP
    Type_SPInstance<VulkanExternalMemoryPool> GetExternalMemoryPool() const {
        return mSPExternalMemoryPool;
    }
#endif

private:
    Type_SPInstance<VulkanInstance> CreateInstance(
        ::std::vector<::std::string> const& requestedLayers,
        ::std::vector<::std::string> const& requestedExtensions);

#ifndef NDEBUG
    Type_SPInstance<VulkanDebugUtils> CreateDebugUtilsMessenger();
#endif

    Type_SPInstance<VulkanSurface> CreateSurface(
        Type_SPInstance<SDLWindow> const& window);

    Type_SPInstance<VulkanPhysicalDevice> PickPhysicalDevice(
        vk::QueueFlags flags);

    Type_SPInstance<VulkanDevice> CreateDevice(
        ::std::vector<::std::string> const& requestedExtensions);

    Type_SPInstance<VulkanMemoryAllocator> CreateVmaAllocator();

#ifdef CUDA_VULKAN_INTEROP
    Type_SPInstance<VulkanExternalMemoryPool> CreateExternalMemoryPool();
#endif

public:
    static void EnableDefaultFeatures();

    static void EnableDynamicRendering();

    static void EnableSynchronization2();

    static void EnableBufferDeviceAddress();

    static void EnableDescriptorIndexing();

private:
    static vk::PhysicalDeviceFeatures sPhysicalDeviceFeatures;
    static vk::PhysicalDeviceVulkan11Features sEnable11Features;
    static vk::PhysicalDeviceVulkan12Features sEnable12Features;
    static vk::PhysicalDeviceVulkan13Features sEnable13Features;

private:
    Type_SPInstance<VulkanInstance> mSPInstance;
#ifndef NDEBUG
    Type_SPInstance<VulkanDebugUtils> mSPDebugUtilsMessenger;
#endif
    Type_SPInstance<VulkanSurface> mSPSurface;
    Type_SPInstance<VulkanPhysicalDevice> mSPPhysicalDevice;
    Type_SPInstance<VulkanDevice> mSPDevice;
    Type_SPInstance<VulkanMemoryAllocator> mSPAllocator;
#ifdef CUDA_VULKAN_INTEROP
    Type_SPInstance<VulkanExternalMemoryPool> mSPExternalMemoryPool;
#endif
};