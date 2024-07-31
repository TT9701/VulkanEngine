#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanHelper.hpp"
#ifndef NDEBUG
#include "VulkanDebugUtils.hpp"
#endif
#include "VulkanBuffer.hpp"
#include "VulkanImage.hpp"
#include "VulkanInstance.hpp"
#include "VulkanMemoryAllocator.hpp"
#include "VulkanPhysicalDevice.hpp"
#include "VulkanSampler.hpp"
#include "VulkanSurface.hpp"

class SDLWindow;

namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}

class VulkanContext {
public:
    VulkanContext(const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
                  ::std::span<::std::string> requestedInstanceLayers     = {},
                  ::std::span<::std::string> requestedInstanceExtensions = {},
                  ::std::span<::std::string> requestedDeviceExtensions   = {});
    ~VulkanContext() = default;
    MOVABLE_ONLY(VulkanContext);

public:
    SharedPtr<VulkanImage> CreateImage2D(
        VmaAllocationCreateFlags flags, vk::Extent3D extent, vk::Format format,
        vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
        void* data = nullptr, VulkanEngine* engine = nullptr,
        uint32_t mipmapLevel = 1, uint32_t arrayLayers = 1);

    SharedPtr<VulkanBuffer> CreatePersistentBuffer(size_t allocByteSize,
                                                   vk::BufferUsageFlags usage);

    SharedPtr<VulkanBuffer> CreateStagingBuffer(size_t allocByteSize,
                                                vk::BufferUsageFlags usage);

#ifdef CUDA_VULKAN_INTEROP
    SharedPtr<CUDA::VulkanExternalImage> CreateExternalImage2D(
        vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
        vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags = {},
        uint32_t mipmapLevels = 1, uint32_t arrayLayers = 1);

    SharedPtr<CUDA::VulkanExternalBuffer> CreateExternalPersistentBuffer(
        size_t allocByteSize, vk::BufferUsageFlags usage);

    SharedPtr<CUDA::VulkanExternalBuffer> CreateExternalStagingBuffer(
        size_t allocByteSize, vk::BufferUsageFlags usage);
#endif

    SharedPtr<VulkanSampler> CreateSampler(
        vk::Filter minFilter, vk::Filter magFilter,
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat,
        float maxLod = 0.0f, bool compareEnable = false,
        vk::CompareOp compareOp = vk::CompareOp::eNever);

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

    VulkanSampler* GetDefaultNearestSampler() const {
        return mDefaultSamplerNearest.get();
    }

    VulkanSampler* GetDefaultLinearSampler() const {
        return mDefaultSamplerLinear.get();
    }

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

    vk::Sampler GetDefaultNearestSamplerHandle() const {
        return mDefaultSamplerNearest->GetHandle();
    }

    vk::Sampler GetDefaultLinearSamplerHandle() const {
        return mDefaultSamplerLinear->GetHandle();
    }

private:
    UniquePtr<VulkanInstance> CreateInstance(
        ::std::span<::std::string> requestedLayers,
        ::std::span<::std::string> requestedExtensions);

#ifndef NDEBUG
    UniquePtr<VulkanDebugUtils> CreateDebugUtilsMessenger();
#endif

    UniquePtr<VulkanSurface> CreateSurface(const SDLWindow* window);

    UniquePtr<VulkanPhysicalDevice> PickPhysicalDevice(vk::QueueFlags flags);

    UniquePtr<VulkanDevice> CreateDevice(
        ::std::span<::std::string> requestedExtensions);

    UniquePtr<VulkanMemoryAllocator> CreateVmaAllocator();

#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<VulkanExternalMemoryPool> CreateExternalMemoryPool();
#endif

    void CreateDefaultSamplers();

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
    UniquePtr<VulkanInstance> mSPInstance;
#ifndef NDEBUG
    UniquePtr<VulkanDebugUtils> mSPDebugUtilsMessenger;
#endif
    UniquePtr<VulkanSurface>         mSPSurface;
    UniquePtr<VulkanPhysicalDevice>  mSPPhysicalDevice;
    UniquePtr<VulkanDevice>          mSPDevice;
    UniquePtr<VulkanMemoryAllocator> mSPAllocator;
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<VulkanExternalMemoryPool> mSPExternalMemoryPool;
#endif
    SharedPtr<VulkanSampler> mDefaultSamplerLinear {};
    SharedPtr<VulkanSampler> mDefaultSamplerNearest {};
};