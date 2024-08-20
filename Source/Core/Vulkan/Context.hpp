#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Device.hpp"
#include "VulkanHelper.hpp"
#ifndef NDEBUG
#include "DebugUtils.hpp"
#endif
#include "Instance.hpp"
#include "MemoryAllocator.hpp"
#include "PhysicalDevice.hpp"
#include "RenderResource.hpp"
#include "Sampler.hpp"
#include "Surface.hpp"
#include "SyncStructures.hpp"

class SDLWindow;

namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}  // namespace CUDA

namespace IntelliDesign_NS::Vulkan::Core {

class Context {
public:
    Context(const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
            ::std::span<::std::string> requestedInstanceLayers = {},
            ::std::span<::std::string> requestedInstanceExtensions = {},
            ::std::span<::std::string> requestedDeviceExtensions = {});
    ~Context() = default;
    MOVABLE_ONLY(Context);

public:
    SharedPtr<RenderResource> CreateTexture2D(vk::Extent3D extent,
                                              vk::Format format,
                                              vk::ImageUsageFlags usage,
                                              uint32_t mipLevels = 1,
                                              uint32_t arraySize = 1,
                                              uint32_t sampleCount = 1);

    SharedPtr<RenderResource> CreateDeviceLocalBuffer(
        size_t allocByteSize, vk::BufferUsageFlags usage);

    SharedPtr<RenderResource> CreateStagingBuffer(
        size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<RenderResource> CreateStorageBuffer(
        size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<RenderResource> CreateIndirectCmdBuffer(size_t allocByteSize);

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

    SharedPtr<Sampler> CreateSampler(
        vk::Filter minFilter, vk::Filter magFilter,
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat,
        float maxLod = 0.0f, bool compareEnable = false,
        vk::CompareOp compareOp = vk::CompareOp::eNever);

    template <class VkCppHandle>
    void SetName(VkCppHandle handle, const char* name);

    template <class VkCppHandle>
    void SetName(VkCppHandle handle, ::std::string_view name);

public:
    // ptrs
    Instance* GetInstance() const;
#ifndef NDEBUG
    DebugUtils* GetDebugMessenger() const;
#endif
    Surface* GetSurface() const;
    PhysicalDevice* GetPhysicalDevice() const;
    Device* GetDevice() const;
    MemoryAllocator* GetVmaAllocator() const;
    TimelineSemaphore* GetTimelineSemphore() const;
#ifdef CUDA_VULKAN_INTEROP
    ExternalMemoryPool* GetExternalMemoryPool() const;
#endif
    Sampler* GetDefaultNearestSampler() const;
    Sampler* GetDefaultLinearSampler() const;

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
    UniquePtr<Instance> CreateInstance(
        ::std::span<::std::string> requestedLayers,
        ::std::span<::std::string> requestedExtensions);

#ifndef NDEBUG
    UniquePtr<DebugUtils> CreateDebugUtilsMessenger();
#endif
    UniquePtr<Surface> CreateSurface(const SDLWindow* window);
    UniquePtr<PhysicalDevice> PickPhysicalDevice(vk::QueueFlags flags);
    UniquePtr<Device> CreateDevice(
        ::std::span<::std::string> requestedExtensions);
    UniquePtr<MemoryAllocator> CreateVmaAllocator();
    UniquePtr<TimelineSemaphore> CreateTimelineSem();
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<ExternalMemoryPool> CreateExternalMemoryPool();
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
    UniquePtr<Instance> mPInstance;
#ifndef NDEBUG
    UniquePtr<DebugUtils> mPDebugUtilsMessenger;
#endif
    UniquePtr<Surface> mPSurface;
    UniquePtr<PhysicalDevice> mPPhysicalDevice;
    UniquePtr<Device> mPDevice;
    UniquePtr<MemoryAllocator> mPAllocator;
    UniquePtr<TimelineSemaphore> mPTimelineSemaphore;
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<ExternalMemoryPool> mPExternalMemoryPool;
#endif
    SharedPtr<Sampler> mDefaultSamplerLinear {};
    SharedPtr<Sampler> mDefaultSamplerNearest {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::Vulkan::Core {

template <class VkCppHandle>
void Context::SetName(VkCppHandle handle, const char* name) {
    mPDevice->SetObjectName(handle, name);
}

template <class VkCppHandle>
void Context::SetName(VkCppHandle handle, std::string_view name) {
    SetName(handle, ::std::string {name}.c_str());
}

}  // namespace IntelliDesign_NS::Vulkan::Core