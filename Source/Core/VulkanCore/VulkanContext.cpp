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
    : mPInstance(
          CreateInstance(requestedInstanceLayers, requestedInstanceExtensions)),
#ifndef NDEBUG
      mPDebugUtilsMessenger(CreateDebugUtilsMessenger()),
#endif
      mPSurface(CreateSurface(window)),
      mPPhysicalDevice(PickPhysicalDevice(requestedQueueFlags)),
      mPDevice(CreateDevice(requestedDeviceExtensions)),
      mPAllocator(CreateVmaAllocator()),
      mPTimelineSemaphore(CreateTimelineSem())
#ifdef CUDA_VULKAN_INTEROP
      ,
      mPExternalMemoryPool(CreateExternalMemoryPool())
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
    return MakeShared<VulkanBuffer>(mPAllocator.get(), allocByteSize, usage,
                                    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
}

SharedPtr<VulkanBuffer> VulkanContext::CreateStagingBuffer(
    size_t allocByteSize) {
    return MakeShared<VulkanBuffer>(
        mPAllocator.get(), allocByteSize, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT);
}

SharedPtr<VulkanBuffer> VulkanContext::CreateUniformBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<VulkanBuffer>(
        mPAllocator.get(), allocByteSize,
        usage | vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

SharedPtr<VulkanBuffer> VulkanContext::CreateStorageBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return CreatePersistentBuffer(
        allocByteSize, usage | vk::BufferUsageFlagBits::eStorageBuffer);
}

SharedPtr<VulkanBuffer> VulkanContext::CreateIndirectCmdBuffer(
    size_t allocByteSize) {
    return CreatePersistentBuffer(allocByteSize,
                                  vk::BufferUsageFlagBits::eIndirectBuffer
                                      | vk::BufferUsageFlagBits::eTransferDst);
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> VulkanContext::CreateExternalImage2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags,
    uint32_t mipmapLevels, uint32_t arrayLayers) {
    return MakeShared<CUDA::VulkanExternalImage>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(),
        mPExternalMemoryPool->GetHandle(), flags, extent, format, usage, aspect,
        mipmapLevels, arrayLayers, vk::ImageType::e2D, vk::ImageViewType::e2D);
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalPersistentBuffer(size_t allocByteSize,
                                              vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mPExternalMemoryPool->GetHandle());
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalStagingBuffer(size_t               allocByteSize,
                                           vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        mPExternalMemoryPool->GetHandle());
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
    return MakeUnique<VulkanDebugUtils>(mPInstance.get());
}
#endif

UniquePtr<VulkanSurface> VulkanContext::CreateSurface(const SDLWindow* window) {
    return MakeUnique<VulkanSurface>(mPInstance.get(), window);
}

UniquePtr<VulkanPhysicalDevice> VulkanContext::PickPhysicalDevice(
    vk::QueueFlags flags) {
    return MakeUnique<VulkanPhysicalDevice>(mPInstance.get(), flags);
}

UniquePtr<VulkanDevice> VulkanContext::CreateDevice(
    ::std::span<::std::string> requestedExtensions) {
    return MakeUnique<VulkanDevice>(
        mPPhysicalDevice.get(), ::std::span<::std::string> {},
        requestedExtensions, &sPhysicalDeviceFeatures, &sEnable11Features);
}

UniquePtr<VulkanMemoryAllocator> VulkanContext::CreateVmaAllocator() {
    return MakeUnique<VulkanMemoryAllocator>(mPPhysicalDevice.get(),
                                             mPDevice.get(), mPInstance.get());
}

UniquePtr<VulkanTimelineSemaphore> VulkanContext::CreateTimelineSem() {
    return MakeUnique<VulkanTimelineSemaphore>(this);
}

#ifdef CUDA_VULKAN_INTEROP
UniquePtr<VulkanExternalMemoryPool> VulkanContext::CreateExternalMemoryPool() {
    return MakeUnique<VulkanExternalMemoryPool>(mPAllocator.get());
}

void VulkanContext::CreateDefaultSamplers() {
    mDefaultSamplerNearest =
        CreateSampler(vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear =
        CreateSampler(vk::Filter::eLinear, vk::Filter::eLinear);
}
#endif

void VulkanContext::EnableDefaultFeatures() {
    sEnable12Features.setRuntimeDescriptorArray(vk::True);
    sEnable11Features.setShaderDrawParameters(vk::True);

    EnableDynamicRendering();
    EnableSynchronization2();
    EnableBufferDeviceAddress();
    EnableDescriptorIndexing();
    EnableTimelineSemaphore();
    EnableMultiDrawIndirect();

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

void VulkanContext::EnableTimelineSemaphore() {
    sEnable12Features.setTimelineSemaphore(vk::True);
}

void VulkanContext::EnableMultiDrawIndirect() {
    sPhysicalDeviceFeatures.setMultiDrawIndirect(vk::True);
    sPhysicalDeviceFeatures.setDrawIndirectFirstInstance(vk::True);
}