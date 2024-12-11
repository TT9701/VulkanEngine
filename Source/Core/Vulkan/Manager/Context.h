#pragma once

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

#include <vulkan/vulkan.hpp>

class SDLWindow;

namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}  // namespace CUDA

namespace IntelliDesign_NS::Vulkan::Core {

class Context {
public:
    Context(SDLWindow& window,
            ::std::span<Type_STLString> requestedInstanceLayers = {},
            ::std::span<Type_STLString> requestedInstanceExtensions = {},
            ::std::span<Type_STLString> requestedDeviceExtensions = {});
    ~Context() = default;
    CLASS_MOVABLE_ONLY(Context);

public:
    void EnableFeatures();

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
    Instance& GetInstance() const;

#ifndef NDEBUG
    DebugUtils& GetDebugMessenger() const;
#endif

    Surface& GetSurface() const;
    PhysicalDevice& GetPhysicalDevice() const;
    Device& GetDevice() const;
    MemoryAllocator& GetVmaAllocator() const;
    TimelineSemaphore& GetTimelineSemphore() const;

#ifdef CUDA_VULKAN_INTEROP
    ExternalMemoryPool* GetExternalMemoryPool() const;
#endif
    Sampler& GetDefaultNearestSampler() const;
    Sampler& GetDefaultLinearSampler() const;

    Queue const& GetPresentQueue() const;
    Queue const& GetGraphicsQueue() const;
    Queue const& GetComputeQueue() const;
    Queue const& GetTransferQueue_ForUpload() const;
    Queue const& GetTransferQueue_ForReadback() const;
    Queue const& GetTransferQueue_ForInternal() const;

private:
    UniquePtr<Instance> CreateInstance(
        ::std::span<Type_STLString> requestedLayers,
        ::std::span<Type_STLString> requestedExtensions);

#ifndef NDEBUG
    UniquePtr<DebugUtils> CreateDebugUtilsMessenger();
#endif
    UniquePtr<Surface> CreateSurface(SDLWindow& window);
    UniquePtr<Device> CreateDevice(
        ::std::span<Type_STLString> requestedExtensions);
    UniquePtr<MemoryAllocator> CreateVmaAllocator();
    UniquePtr<TimelineSemaphore> CreateTimelineSem();
#ifdef CUDA_VULKAN_INTEROP
    UniquePtr<ExternalMemoryPool> CreateExternalMemoryPool();
#endif
    void CreateDefaultSamplers();

private:
    UniquePtr<Instance> mInstance;
#ifndef NDEBUG
    UniquePtr<DebugUtils> mDebugUtilsMessenger;
#endif
    UniquePtr<Surface> mSurface;
    PhysicalDevice& mPhysicalDevice;
    UniquePtr<Device> mDevice;
    UniquePtr<MemoryAllocator> mAllocator;
    UniquePtr<TimelineSemaphore> mTimelineSemaphore;
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
    mDevice->SetObjectName(handle, name);
}

template <class VkCppHandle>
void Context::SetName(VkCppHandle handle, std::string_view name) {
    SetName(handle, Type_STLString {name}.c_str());
}

}  // namespace IntelliDesign_NS::Vulkan::Core