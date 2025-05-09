﻿#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Device.h"
#ifndef NDEBUG
#include "Core/Vulkan/Native/DebugUtils.h"
#endif
#include "Core/Utilities/Profiler.h"
#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/Instance.h"
#include "Core/Vulkan/Native/MemoryAllocator.h"
#include "Core/Vulkan/Native/PhysicalDevice.h"
#include "Core/Vulkan/Native/RenderResource.h"
#include "Core/Vulkan/Native/Sampler.h"
#include "Core/Vulkan/Native/Surface.h"
#include "Core/Vulkan/Native/SyncStructures.h"

class SDLWindow;

#ifdef CUDA_VULKAN_INTEROP
namespace CUDA {
class VulkanExternalImage;
class VulkanExternalBuffer;
}  // namespace CUDA
#endif

namespace IntelliDesign_NS::Vulkan::Core {

enum class QueueType {
    Present,
    Graphics,
    Compute,
    Async_Compute,
    Transfer,
    Async_Transfer_Upload,
    Async_Transfer_Readback
};

/**
 * @brief 
 */
class VulkanContext {
public:
    /**
     * @brief 
     * @param window 
     * @param requestedInstanceLayers 
     * @param requestedInstanceExtensions 
     * @param requestedDeviceExtensions 
     */
    VulkanContext(SDLWindow& window,
                  ::std::span<Type_STLString> requestedInstanceLayers = {},
                  ::std::span<Type_STLString> requestedInstanceExtensions = {},
                  ::std::span<Type_STLString> requestedDeviceExtensions = {});

    /**
     * @brief 
     */
    ~VulkanContext() = default;

    /**
     *
     */
    CLASS_MOVABLE_ONLY(VulkanContext);

    /**
     * @brief RAII command buffer
     */
    struct CmdToBegin {
        CmdToBegin(Device& device, vk::CommandBuffer cmd, vk::CommandPool pool,
                   vk::Queue queue, vk::Semaphore signal);
        ~CmdToBegin();

        vk::CommandBuffer GetHandle() const { return mHandle; }

        vk::CommandBuffer const* operator->() const;

        Device& mDevice;
        vk::CommandBuffer mHandle;
        vk::CommandPool mPool;
        vk::Queue mQueue;
        vk::Semaphore mSem;
    };

public:
    /**
     * @brief 
     */
    void EnableFeatures();

    /**
     * @brief 
     * @param queue 
     * @param signal 
     * @param level 
     * @return 
     */
    CmdToBegin CreateCmdBufToBegin(
        Queue const& queue, vk::Semaphore signal = VK_NULL_HANDLE,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;

    SharedPtr<Texture> CreateTexture2D(const char* name, vk::Extent3D extent,
                                       vk::Format format,
                                       vk::ImageUsageFlags usage,
                                       uint32_t mipLevels = 1,
                                       uint32_t arraySize = 1,
                                       uint32_t sampleCount = 1);

    SharedPtr<RenderResource> CreateDeviceLocalBufferResource(
        const char* name, size_t allocByteSize, vk::BufferUsageFlags usage,
        size_t stride = 1);

    SharedPtr<Buffer> CreateDeviceLocalBuffer(const char* name,
                                              size_t allocByteSize,
                                              vk::BufferUsageFlags usage,
                                              size_t stride = 1);

    template <class T>
    SharedPtr<Buffer> CreateDeviceLocalBuffer_WithData(
        const char* name, vk::BufferUsageFlags usage,
        Type_STLVector<T> const& data);

    SharedPtr<Buffer> CreateStagingBuffer(
        const char* name, size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0,
        size_t stride = 1);

    SharedPtr<Buffer> CreateReadBackBuffer(
        const char* name, size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0,
        size_t stride = 1);

    SharedPtr<Buffer> CreateStorageBuffer(
        const char* name, size_t allocByteSize,
        vk::BufferUsageFlags usage = (vk::BufferUsageFlagBits)0,
        size_t stride = 1);

    template <class T>
    SharedPtr<StructuredBuffer<T>> CreateIndirectCmdBuffer(const char* name,
                                                           uint32_t count);

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

    /**
     * @brief 
     * @tparam VkCppHandle 
     * @param handle 
     * @param name 
     */
    template <class VkCppHandle>
    void SetName(VkCppHandle handle, const char* name);

    /**
     * @brief 
     * @tparam VkCppHandle 
     * @param handle 
     * @param name 
     */
    template <class VkCppHandle>
    void SetName(VkCppHandle handle, ::std::string_view name);

public:
    /**
     * @brief 
     * @return 
     */
    Instance& GetInstance() const;

#ifndef NDEBUG
    /**
     * @brief 
     * @return 
     */
    DebugUtils& GetDebugMessenger() const;
#endif

    /**
     * @brief 
     * @return 
     */
    Surface& GetSurface() const;

    /**
     * @brief 
     * @return 
     */
    PhysicalDevice& GetPhysicalDevice() const;

    /**
     * @brief 
     * @return 
     */
    Device& GetDevice() const;

    /**
     * @brief 
     * @return 
     */
    MemoryAllocator& GetVmaAllocator() const;

    TimelineSemaphore& GetTimelineSemphore() const;

#ifdef CUDA_VULKAN_INTEROP
    ExternalMemoryPool* GetExternalMemoryPool() const;
#endif
    Sampler& GetDefaultNearestSampler() const;
    Sampler& GetDefaultLinearSampler() const;

    /**
     * @brief 获取特定用法的 queue
     * @param type queue 的类型
     * @param highPriority 是否需要高优先级 queue，仅对 Present、Graphics、
     *        Compute_Prepare、Transfer_Prepare 用法生效
     * @return 特定用法的 queue 的常量引用
     */
    Queue const& GetQueue(QueueType type = QueueType::Graphics,
                          bool highPriority = true) const;

    /**
     * @brief 
     * @return 
     */
    FencePool& GetFencePool() const;

    /**
     * @brief 
     * @return 
     */
    CommandPool& GetCommandPool() const;

    /**
     * @brief 
     * @return 
     */
    IntelliDesign_NS::Core::TracyProfiler& GetProfiler() const;

private:
    /**
     * @brief 
     * @param requestedLayers 
     * @param requestedExtensions 
     * @return 
     */
    UniquePtr<Instance> CreateInstance(
        ::std::span<Type_STLString> requestedLayers,
        ::std::span<Type_STLString> requestedExtensions);

#ifndef NDEBUG
    /**
     * @brief 
     * @return 
     */
    UniquePtr<DebugUtils> CreateDebugUtilsMessenger();
#endif

    /**
     * @brief 
     * @param window 
     * @return 
     */
    UniquePtr<Surface> CreateSurface(SDLWindow& window);

    /**
     * @brief 
     * @param requestedExtensions 
     * @return 
     */
    UniquePtr<Device> CreateDevice(
        ::std::span<Type_STLString> requestedExtensions);

    /**
     * @brief 
     * @return 
     */
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

    UniquePtr<FencePool> mFencePool;
    UniquePtr<CommandPool> mCommandPool;

    UniquePtr<IntelliDesign_NS::Core::TracyProfiler> mProfiler;
};

}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::Vulkan::Core {

template <class T>
SharedPtr<Buffer> VulkanContext::CreateDeviceLocalBuffer_WithData(
    const char* name, vk::BufferUsageFlags usage,
    Type_STLVector<T> const& data) {
    VE_ASSERT(!data.empty(), "No data was set.")

    size_t const dataSize = sizeof(T) * data.size();

    auto ptr = CreateDeviceLocalBuffer(name, dataSize, usage, sizeof(T));

    auto staging = CreateStagingBuffer("", dataSize);

    memcpy(staging->GetMapPtr(), data.data(), dataSize);

    {
        auto cmd = CreateCmdBufToBegin(GetQueue(QueueType::Transfer));
        vk::BufferCopy cmdBufCopy {};
        cmdBufCopy.setSize(dataSize);
        cmd->copyBuffer(staging->GetHandle(), ptr->GetHandle(), cmdBufCopy);
    }

    return ptr;
}

template <class T>
SharedPtr<StructuredBuffer<T>> VulkanContext::CreateIndirectCmdBuffer(
    const char* name, uint32_t count) {
    auto ptr = MakeShared<StructuredBuffer<T>>(
        *this, count,
        vk::BufferUsageFlagBits::eIndirectBuffer
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::DeviceLocal);

    ptr->SetName(name);
    return ptr;
}

template <class VkCppHandle>
void VulkanContext::SetName(VkCppHandle handle, const char* name) {
    mDevice->SetObjectName(handle, name);
}

}  // namespace IntelliDesign_NS::Vulkan::Core