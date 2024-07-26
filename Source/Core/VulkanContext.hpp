#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "VulkanHelper.hpp"

class VulkanInstance;
class VulkanDebugUtils;
class VulkanSurface;
class VulkanPhysicalDevice;
class VulkanDevice;
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
    Type_SPInstance<VulkanInstance> GetInstancePtr() const {
        return mSPInstance;
    }

#ifdef DEBUG
    Type_SPInstance<VulkanDebugUtils> GetDebugMessengerPtr() const {
        return mSPDebugUtilsMessenger;
    }
#endif

    Type_SPInstance<VulkanSurface> GetSurfacePtr() const { return mSPSurface; }

    Type_SPInstance<VulkanPhysicalDevice> GetPhysicalDevicePtr() const {
        return mSPPhysicalDevice;
    }

    Type_SPInstance<VulkanDevice> GetDevicePtr() const { return mSPDevice; }

    vk::Instance const& GetInstanceHandle() const;

#ifdef DEBUG
    vk::DebugUtilsMessengerEXT const& GetDebugMessengerHandle() const;
#endif

    VkSurfaceKHR const& GetSurfaceHandle() const;

    vk::PhysicalDevice const& GetPhysicalDeviceHandle() const;

    vk::Device const& GetDeviceHandle() const;

public:
private:
    Type_SPInstance<VulkanInstance> CreateInstance(
        ::std::vector<::std::string> const& requestedLayers,
        ::std::vector<::std::string> const& requestedExtensions);

#ifdef DEBUG
    Type_SPInstance<VulkanDebugUtils> CreateDebugUtilsMessenger();
#endif

    Type_SPInstance<VulkanSurface> CreateSurface(
        Type_SPInstance<SDLWindow> const& window);

    Type_SPInstance<VulkanPhysicalDevice> PickPhysicalDevice(
        vk::QueueFlags flags);

    Type_SPInstance<VulkanDevice> CreateDevice(
        ::std::vector<::std::string> const& requestedExtensions);

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
#ifdef DEBUG
    Type_SPInstance<VulkanDebugUtils> mSPDebugUtilsMessenger;
#endif
    Type_SPInstance<VulkanSurface> mSPSurface;
    Type_SPInstance<VulkanPhysicalDevice> mSPPhysicalDevice;
    Type_SPInstance<VulkanDevice> mSPDevice;
};