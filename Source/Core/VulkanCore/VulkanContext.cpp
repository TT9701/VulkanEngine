#include "VulkanContext.hpp"

#include "CUDA/CUDAVulkan.h"

vk::PhysicalDeviceFeatures         VulkanContext::sPhysicalDeviceFeatures {};
vk::PhysicalDeviceVulkan11Features VulkanContext::sEnable11Features {};
vk::PhysicalDeviceVulkan12Features VulkanContext::sEnable12Features {};
vk::PhysicalDeviceVulkan13Features VulkanContext::sEnable13Features {};

VulkanContext::VulkanContext(
    const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
    ::std::span<::std::string> requestedInstanceLayers,
    ::std::span<::std::string> requestedInstanceExtensions,
    ::std::span<::std::string> requestedDeviceExtensions)
    : mSPInstance(
          CreateInstance(requestedInstanceLayers, requestedInstanceExtensions)),
#ifndef NDEBUG
      mSPDebugUtilsMessenger(CreateDebugUtilsMessenger()),
#endif
      mSPSurface(CreateSurface(window)),
      mSPPhysicalDevice(PickPhysicalDevice(requestedQueueFlags)),
      mSPDevice(CreateDevice(requestedDeviceExtensions)),
      mSPAllocator(CreateVmaAllocator()),
#ifdef CUDA_VULKAN_INTEROP
      mSPExternalMemoryPool(CreateExternalMemoryPool())
#endif
{
    CreateDefaultSamplers();
}

SharedPtr<VulkanImage> VulkanContext::CreateImage2D(
    VmaAllocationCreateFlags flags, vk::Extent3D extent, vk::Format format,
    vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect, void* data,
    VulkanEngine* engine, uint32_t mipmapLevel, uint32_t arrayLayers) {
    return MakeShared<VulkanImage>(this, flags, extent, format, usage, aspect,
                                   data, engine, mipmapLevel, arrayLayers,
                                   vk::ImageType::e2D, vk::ImageViewType::e2D);
}

SharedPtr<VulkanBuffer> VulkanContext::CreatePersistentBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<VulkanBuffer>(mSPAllocator.get(), allocByteSize, usage,
                                    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
}

SharedPtr<VulkanBuffer> VulkanContext::CreateStagingBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<VulkanBuffer>(
        mSPAllocator.get(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT);
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> VulkanContext::CreateExternalImage2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags,
    uint32_t mipmapLevels, uint32_t arrayLayers) {
    return MakeShared<CUDA::VulkanExternalImage>(
        mSPDevice->GetHandle(), mSPAllocator->GetHandle(),
        mSPExternalMemoryPool->GetHandle(), flags, extent, format, usage,
        aspect, mipmapLevels, arrayLayers, vk::ImageType::e2D,
        vk::ImageViewType::e2D);
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalPersistentBuffer(size_t allocByteSize,
                                              vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mSPDevice->GetHandle(), mSPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mSPExternalMemoryPool->GetHandle());
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalStagingBuffer(size_t               allocByteSize,
                                           vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mSPDevice->GetHandle(), mSPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        mSPExternalMemoryPool->GetHandle());
}
#endif

SharedPtr<VulkanSampler> VulkanContext::CreateSampler(
    vk::Filter minFilter, vk::Filter magFilter,
    vk::SamplerAddressMode addressModeU, vk::SamplerAddressMode addressModeV,
    vk::SamplerAddressMode addressModeW, float maxLod, bool compareEnable,
    vk::CompareOp compareOp) {
    return MakeShared<VulkanSampler>(this, minFilter, magFilter, addressModeU,
                                     addressModeV, addressModeW, maxLod,
                                     compareEnable, compareOp);
}

UniquePtr<VulkanInstance> VulkanContext::CreateInstance(
    ::std::span<::std::string> requestedLayers,
    ::std::span<::std::string> requestedExtensions) {
    return MakeUnique<VulkanInstance>(requestedLayers, requestedExtensions);
}

#ifndef NDEBUG
UniquePtr<VulkanDebugUtils> VulkanContext::CreateDebugUtilsMessenger() {
    return MakeUnique<VulkanDebugUtils>(mSPInstance.get());
}
#endif

UniquePtr<VulkanSurface> VulkanContext::CreateSurface(const SDLWindow* window) {
    return MakeUnique<VulkanSurface>(mSPInstance.get(), window);
}

UniquePtr<VulkanPhysicalDevice> VulkanContext::PickPhysicalDevice(
    vk::QueueFlags flags) {
    return MakeUnique<VulkanPhysicalDevice>(mSPInstance.get(), flags);
}

UniquePtr<VulkanDevice> VulkanContext::CreateDevice(
    ::std::span<::std::string> requestedExtensions) {
    return MakeUnique<VulkanDevice>(
        mSPPhysicalDevice.get(), ::std::span<::std::string> {},
        requestedExtensions, &sPhysicalDeviceFeatures, &sEnable11Features);
}

UniquePtr<VulkanMemoryAllocator> VulkanContext::CreateVmaAllocator() {
    return MakeUnique<VulkanMemoryAllocator>(
        mSPPhysicalDevice.get(), mSPDevice.get(), mSPInstance.get());
}

#ifdef CUDA_VULKAN_INTEROP
UniquePtr<VulkanExternalMemoryPool> VulkanContext::CreateExternalMemoryPool() {
    return MakeUnique<VulkanExternalMemoryPool>(mSPAllocator.get());
}

void VulkanContext::CreateDefaultSamplers() {
    mDefaultSamplerNearest =
        CreateSampler(vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear =
        CreateSampler(vk::Filter::eLinear, vk::Filter::eLinear);
}
#endif

void VulkanContext::EnableDefaultFeatures() {
    EnableDynamicRendering();
    EnableSynchronization2();
    EnableBufferDeviceAddress();
    EnableDescriptorIndexing();

    sEnable11Features.setPNext(&sEnable12Features);
    sEnable12Features.setPNext(&sEnable13Features);
}

void VulkanContext::EnableDynamicRendering() {
    sEnable13Features.setDynamicRendering(vk::True);
}

void VulkanContext::EnableSynchronization2() {
    sEnable13Features.setSynchronization2(vk::True);
}

void VulkanContext::EnableBufferDeviceAddress() {
    sEnable12Features.setBufferDeviceAddress(vk::True);
}

void VulkanContext::EnableDescriptorIndexing() {
    sEnable12Features.setDescriptorIndexing(vk::True);
}