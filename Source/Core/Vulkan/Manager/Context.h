#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Device.h"
#ifndef NDEBUG
#include "Core/Vulkan/Native/DebugUtils.h"
#endif
#include "Core/Vulkan/Native/Instance.h"
#include "Core/Vulkan/Native/MemoryAllocator.h"
#include "Core/Vulkan/Native/PhysicalDevice.h"
#include "Core/Vulkan/Native/RenderResource.h"
#include "Core/Vulkan/Native/Sampler.h"
#include "Core/Vulkan/Native/Surface.h"
#include "Core/Vulkan/Native/SyncStructures.h"

class SDLWindow;

namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}  // namespace CUDA

namespace IntelliDesign_NS::Vulkan::Core {

class Context {
public:
    Context(const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
            ::std::span<Type_STLString> requestedInstanceLayers = {},
            ::std::span<Type_STLString> requestedInstanceExtensions = {},
            ::std::span<Type_STLString> requestedDeviceExtensions = {});
    ~Context() = default;
    CLASS_MOVABLE_ONLY(Context);

public:
    SharedPtr<Texture> CreateTexture2D(const char* name, vk::Extent3D extent,
                                       vk::Format format,
                                       vk::ImageUsageFlags usage,
                                       uint32_t mipLevels = 1,
                                       uint32_t arraySize = 1,
                                       uint32_t sampleCount = 1);

    SharedPtr<RenderResource> CreateDeviceLocalBufferResource(
        const char* name, size_t allocByteSize, vk::BufferUsageFlags usage);

    SharedPtr<Buffer> CreateDeviceLocalBuffer(const char* name,
                                              size_t allocByteSize,
                                              vk::BufferUsageFlags usage);

    SharedPtr<Buffer> CreateStagingBuffer(
        const char* name, size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<Buffer> CreateStorageBuffer(
        const char* name, size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0);

    SharedPtr<Buffer> CreateIndirectCmdBuffer(const char* name,
                                              size_t allocByteSize);

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

    vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& GetDescBufProps()
        const;
    vk::PhysicalDeviceDescriptorIndexingProperties const& GetDescIndexingProps()
        const;

private:
    UniquePtr<Instance> CreateInstance(
        ::std::span<Type_STLString> requestedLayers,
        ::std::span<Type_STLString> requestedExtensions);

#ifndef NDEBUG
    UniquePtr<DebugUtils> CreateDebugUtilsMessenger();
#endif
    UniquePtr<Surface> CreateSurface(const SDLWindow* window);
    UniquePtr<PhysicalDevice> PickPhysicalDevice(vk::QueueFlags flags);
    UniquePtr<Device> CreateDevice(
        ::std::span<Type_STLString> requestedExtensions);
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
    static vk::PhysicalDeviceMeshShaderFeaturesEXT sEnableMeshShaderFeaturesExt;
    static vk::PhysicalDeviceDescriptorBufferFeaturesEXT
        sEnableDescriptorBufferFeaturesExt;
    static vk::PhysicalDeviceMaintenance6FeaturesKHR sEnableMaintenance6KHR;

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

    vk::PhysicalDeviceDescriptorBufferPropertiesEXT mDescBufProps;
    vk::PhysicalDeviceDescriptorIndexingProperties mDescIndexingProps;
};

}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::Vulkan::Core {

template <class VkCppHandle>
void Context::SetName(VkCppHandle handle, const char* name) {
    mPDevice->SetObjectName(handle, name);
}

template <class VkCppHandle>
void Context::SetName(VkCppHandle handle, std::string_view name) {
    SetName(handle, Type_STLString {name}.c_str());
}

}  // namespace IntelliDesign_NS::Vulkan::Core