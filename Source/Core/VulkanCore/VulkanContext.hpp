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
#include "VulkanSyncStructures.hpp"

class SDLWindow;

namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}  // namespace CUDA

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

    SharedPtr<VulkanBuffer> CreateStagingBuffer(size_t allocByteSize);

    SharedPtr<VulkanBuffer> CreateUniformBuffer(
        size_t               allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<VulkanBuffer> CreateStorageBuffer(
        size_t               allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<VulkanBuffer> CreateIndirectCmdBuffer(size_t allocByteSize);

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
    VulkanInstance* GetInstance() const { return mPInstance.get(); }

#ifndef NDEBUG
    VulkanDebugUtils* GetDebugMessenger() const {
        return mPDebugUtilsMessenger.get();
    }
#endif

    VulkanSurface* GetSurface() const { return mPSurface.get(); }

    VulkanPhysicalDevice* GetPhysicalDevice() const {
        return mPPhysicalDevice.get();
    }

    VulkanDevice* GetDevice() const { return mPDevice.get(); }

    VulkanMemoryAllocator* GetVmaAllocator() const { return mPAllocator.get(); }

    VulkanTimelineSemaphore* GetTimelineSemphore() const {
        return mPTimelineSemaphore.get();
    }

#ifdef CUDA_VULKAN_INTEROP
    VulkanExternalMemoryPool* GetExternalMemoryPool() const {
        return mPExternalMemoryPool.get();
    }
#endif

    VulkanSampler* GetDefaultNearestSampler() const {
        return mDefaultSamplerNearest.get();
    }

    VulkanSampler* GetDefaultLinearSampler() const {
        return mDefaultSamplerLinear.get();
    }

    vk::Instance GetInstanceHandle() const { return mPInstance->GetHandle(); }

#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT GetDebugMessengerHandle() const {
        return mPDebugUtilsMessenger->GetHandle();
    }
#endif

    vk::SurfaceKHR GetSurfaceHandle() const { return mPSurface->GetHandle(); }

    vk::PhysicalDevice GetPhysicalDeviceHandle() const {
        return mPPhysicalDevice->GetHandle();
    }

    vk::Device GetDeviceHandle() const { return mPDevice->GetHandle(); }

    VmaAllocator GetVmaAllocatorHandle() const {
        return mPAllocator->GetHandle();
    }

    vk::Semaphore GetTimelineSemaphoreHandle() const {
        return mPTimelineSemaphore->GetHandle();
    }

#ifdef CUDA_VULKAN_INTEROP
    VmaPool GetExternalMemoryPoolHandle() const {
        return mPExternalMemoryPool->GetHandle();
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

    UniquePtr<VulkanTimelineSemaphore> CreateTimelineSem();

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
    static void EnableTimelineSemaphore();
    static void EnableMultiDrawIndirect();

private:
    static vk::PhysicalDeviceFeatures         sPhysicalDeviceFeatures;
    static vk::PhysicalDeviceVulkan11Features sEnable11Features;
    static vk::PhysicalDeviceVulkan12Features sEnable12Features;
    static vk::PhysicalDeviceVulkan13Features sEnable13Features;

private:
    UniquePtr<VulkanInstance> mPInstance;
#ifndef NDEBUG
    UniquePtr<VulkanDebugUtils> mPDebugUtilsMessenger;
#endif
    UniquePtr<VulkanSurface>           mPSurface;
    UniquePtr<VulkanPhysicalDevice>    mPPhysicalDevice;
    UniquePtr<VulkanDevice>            mPDevice;
    UniquePtr<VulkanMemoryAllocator>   mPAllocator;
    UniquePtr<VulkanTimelineSemaphore> mPTimelineSemaphore;
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<VulkanExternalMemoryPool> mPExternalMemoryPool;
#endif
    SharedPtr<VulkanSampler> mDefaultSamplerLinear {};
    SharedPtr<VulkanSampler> mDefaultSamplerNearest {};
};