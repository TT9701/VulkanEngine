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
#include "VulkanResource.h"
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
                  ::std::span<::std::string> requestedInstanceLayers = {},
                  ::std::span<::std::string> requestedInstanceExtensions = {},
                  ::std::span<::std::string> requestedDeviceExtensions = {});
    ~VulkanContext() = default;
    MOVABLE_ONLY(VulkanContext);

public:
    SharedPtr<VulkanResource> CreateTexture2D(vk::Extent3D extent,
                                              vk::Format format,
                                              vk::ImageUsageFlags usage,
                                              uint32_t mipLevels = 1,
                                              uint32_t arraySize = 1,
                                              uint32_t sampleCount = 1);

    SharedPtr<VulkanResource> CreateDeviceLocalBuffer(
        size_t allocByteSize, vk::BufferUsageFlags usage);

    SharedPtr<VulkanResource> CreateStagingBuffer(
        size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<VulkanResource> CreateStorageBuffer(
        size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<VulkanResource> CreateIndirectCmdBuffer(size_t allocByteSize);

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

    template <class VkCppHandle>
    void SetName(VkCppHandle handle, const char* name);

public:
    // ptrs
    VulkanInstance* GetInstance() const;
#ifndef NDEBUG
    VulkanDebugUtils* GetDebugMessenger() const;
#endif
    VulkanSurface* GetSurface() const;
    VulkanPhysicalDevice* GetPhysicalDevice() const;
    VulkanDevice* GetDevice() const;
    VulkanMemoryAllocator* GetVmaAllocator() const;
    VulkanTimelineSemaphore* GetTimelineSemphore() const;
#ifdef CUDA_VULKAN_INTEROP
    VulkanExternalMemoryPool* GetExternalMemoryPool() const;
#endif
    VulkanSampler* GetDefaultNearestSampler() const;
    VulkanSampler* GetDefaultLinearSampler() const;

    // handles
    vk::Instance GetInstanceHandle() const;
#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT GetDebugMessengerHandle() const;
#endif
    vk::SurfaceKHR GetSurfaceHandle() const;
    vk::PhysicalDevice GetPhysicalDeviceHandle() const;
    vk::Device GetDeviceHandle() const;
    VmaAllocator GetVmaAllocatorHandle() const;
    vk::Semaphore GetTimelineSemaphoreHandle() const;
#ifdef CUDA_VULKAN_INTEROP
    VmaPool GetExternalMemoryPoolHandle() const;
#endif
    vk::Sampler GetDefaultNearestSamplerHandle() const;
    vk::Sampler GetDefaultLinearSamplerHandle() const;

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
    static vk::PhysicalDeviceFeatures sPhysicalDeviceFeatures;
    static vk::PhysicalDeviceVulkan11Features sEnable11Features;
    static vk::PhysicalDeviceVulkan12Features sEnable12Features;
    static vk::PhysicalDeviceVulkan13Features sEnable13Features;

private:
    UniquePtr<VulkanInstance> mPInstance;
#ifndef NDEBUG
    UniquePtr<VulkanDebugUtils> mPDebugUtilsMessenger;
#endif
    UniquePtr<VulkanSurface> mPSurface;
    UniquePtr<VulkanPhysicalDevice> mPPhysicalDevice;
    UniquePtr<VulkanDevice> mPDevice;
    UniquePtr<VulkanMemoryAllocator> mPAllocator;
    UniquePtr<VulkanTimelineSemaphore> mPTimelineSemaphore;
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<VulkanExternalMemoryPool> mPExternalMemoryPool;
#endif
    SharedPtr<VulkanSampler> mDefaultSamplerLinear {};
    SharedPtr<VulkanSampler> mDefaultSamplerNearest {};
};

template <class VkCppHandle>
void VulkanContext::SetName(VkCppHandle handle, const char* name) {
    mPDevice->SetObjectName(handle, name);
}