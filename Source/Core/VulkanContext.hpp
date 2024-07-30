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
public:
    VulkanContext(
        const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
        ::std::span<::std::string> requestedInstanceLayers     = {},
        ::std::span<::std::string> requestedInstanceExtensions = {},
        ::std::span<::std::string> requestedDeviceExtensions   = {});
    ~VulkanContext() = default;
    MOVABLE_ONLY(VulkanContext);

public:
    VulkanInstance* GetInstance() const { return mSPInstance.get(); }

#ifndef NDEBUG
    VulkanDebugUtils* GetDebugMessenger() const {
        return mSPDebugUtilsMessenger.get();
    }
#endif

    VulkanSurface* GetSurface() const { return mSPSurface.get(); }

    VulkanPhysicalDevice* GetPhysicalDevice() const {
        return mSPPhysicalDevice.get();
    }

    VulkanDevice* GetDevice() const { return mSPDevice.get(); }

    VulkanMemoryAllocator* GetVmaAllocator() const {
        return mSPAllocator.get();
    }

#ifdef CUDA_VULKAN_INTEROP
    VulkanExternalMemoryPool* GetExternalMemoryPool() const {
        return mSPExternalMemoryPool.get();
    }
#endif

    vk::Instance GetInstanceHandle() const { return mSPInstance->GetHandle(); }

#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT GetDebugMessengerHandle() const {
        return mSPDebugUtilsMessenger->GetHandle();
    }
#endif

    vk::SurfaceKHR GetSurfaceHandle() const { return mSPSurface->GetHandle(); }

    vk::PhysicalDevice GetPhysicalDeviceHandle() const {
        return mSPPhysicalDevice->GetHandle();
    }

    vk::Device GetDeviceHandle() const { return mSPDevice->GetHandle(); }

    VmaAllocator GetVmaAllocatorHandle() const {
        return mSPAllocator->GetHandle();
    }

#ifdef CUDA_VULKAN_INTEROP
    VmaPool GetExternalMemoryPoolHandle() const {
        return mSPExternalMemoryPool->GetHandle();
    }
#endif

private:
    SharedPtr<VulkanInstance> CreateInstance(
        ::std::span<::std::string> requestedLayers,
        ::std::span<::std::string> requestedExtensions);

#ifndef NDEBUG
    SharedPtr<VulkanDebugUtils> CreateDebugUtilsMessenger();
#endif

    SharedPtr<VulkanSurface> CreateSurface(const SDLWindow* window);

    SharedPtr<VulkanPhysicalDevice> PickPhysicalDevice(vk::QueueFlags flags);

    SharedPtr<VulkanDevice> CreateDevice(
        ::std::span<::std::string> requestedExtensions);

    SharedPtr<VulkanMemoryAllocator> CreateVmaAllocator();

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<VulkanExternalMemoryPool> CreateExternalMemoryPool();
#endif

public:
    static void EnableDefaultFeatures();

    static void EnableDynamicRendering();

    static void EnableSynchronization2();

    static void EnableBufferDeviceAddress();

    static void EnableDescriptorIndexing();

private:
    static vk::PhysicalDeviceFeatures         sPhysicalDeviceFeatures;
    static vk::PhysicalDeviceVulkan11Features sEnable11Features;
    static vk::PhysicalDeviceVulkan12Features sEnable12Features;
    static vk::PhysicalDeviceVulkan13Features sEnable13Features;

private:
    SharedPtr<VulkanInstance> mSPInstance;
#ifndef NDEBUG
    SharedPtr<VulkanDebugUtils> mSPDebugUtilsMessenger;
#endif
    SharedPtr<VulkanSurface>         mSPSurface;
    SharedPtr<VulkanPhysicalDevice>  mSPPhysicalDevice;
    SharedPtr<VulkanDevice>          mSPDevice;
    SharedPtr<VulkanMemoryAllocator> mSPAllocator;
#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<VulkanExternalMemoryPool> mSPExternalMemoryPool;
#endif
};