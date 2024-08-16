#include "VulkanContext.hpp"

#include "CUDA/CUDAVulkan.h"

vk::PhysicalDeviceFeatures VulkanContext::sPhysicalDeviceFeatures {};
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
    mPDevice->SetObjectName(mPInstance->GetHandle(), "Default Instance");
    mPDevice->SetObjectName(mPSurface->GetHandle(), "Default Surface");
    mPDevice->SetObjectName(
        mPPhysicalDevice->GetHandle(),
        mPPhysicalDevice->GetHandle().getProperties2().properties.deviceName);
    mPDevice->SetObjectName(mPDevice->GetHandle(), "Default Device");
    mPDevice->SetObjectName(mPTimelineSemaphore->GetHandle(),
                            "Main Timeline Semaphore");

    CreateDefaultSamplers();

    mPDevice->SetObjectName(mDefaultSamplerLinear->GetHandle(),
                            "Default Linear Sampler");
    mPDevice->SetObjectName(mDefaultSamplerNearest->GetHandle(),
                            "Default Nearest Sampler");

    mPDevice->SetObjectName(mPTimelineSemaphore->GetHandle(),
                            "Main Timeline Semaphore");
}

SharedPtr<VulkanResource> VulkanContext::CreateTexture2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    uint32_t mipLevels, uint32_t arraySize, uint32_t sampleCount) {
    return MakeShared<VulkanResource>(
        mPDevice.get(), mPAllocator.get(), VulkanResource::Type::Texture2D,
        format, extent, usage, mipLevels, arraySize, sampleCount);
}

SharedPtr<VulkanResource> VulkanContext::CreateDeviceLocalBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<VulkanResource>(
        mPDevice.get(), mPAllocator.get(), VulkanResource::Type::Buffer,
        allocByteSize, usage, BufferMemoryType::DeviceLocal);
}

SharedPtr<VulkanResource> VulkanContext::CreateStagingBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<VulkanResource>(
        mPDevice.get(), mPAllocator.get(), VulkanResource::Type::Buffer,
        allocByteSize, usage | vk::BufferUsageFlagBits::eTransferSrc,
        BufferMemoryType::Staging);
}

SharedPtr<VulkanResource> VulkanContext::CreateStorageBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return CreateDeviceLocalBuffer(
        allocByteSize, usage | vk::BufferUsageFlagBits::eStorageBuffer);
}

SharedPtr<VulkanResource> VulkanContext::CreateIndirectCmdBuffer(
    size_t allocByteSize) {
    return CreateDeviceLocalBuffer(allocByteSize,
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
VulkanContext::CreateExternalStagingBuffer(size_t allocByteSize,
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

VulkanInstance* VulkanContext::GetInstance() const {
    return mPInstance.get();
}

#ifndef NDEBUG
VulkanDebugUtils* VulkanContext::GetDebugMessenger() const {
    return mPDebugUtilsMessenger.get();
}
#endif

VulkanSurface* VulkanContext::GetSurface() const {
    return mPSurface.get();
}

VulkanPhysicalDevice* VulkanContext::GetPhysicalDevice() const {
    return mPPhysicalDevice.get();
}

VulkanDevice* VulkanContext::GetDevice() const {
    return mPDevice.get();
}

VulkanMemoryAllocator* VulkanContext::GetVmaAllocator() const {
    return mPAllocator.get();
}

VulkanTimelineSemaphore* VulkanContext::GetTimelineSemphore() const {
    return mPTimelineSemaphore.get();
}

#ifdef CUDA_VULKAN_INTEROP
VulkanExternalMemoryPool* VulkanContext::GetExternalMemoryPool() const {
    return mPExternalMemoryPool.get();
}
#endif

VulkanSampler* VulkanContext::GetDefaultNearestSampler() const {
    return mDefaultSamplerNearest.get();
}

VulkanSampler* VulkanContext::GetDefaultLinearSampler() const {
    return mDefaultSamplerLinear.get();
}

vk::Instance VulkanContext::GetInstanceHandle() const {
    return mPInstance->GetHandle();
}

#ifndef NDEBUG
vk::DebugUtilsMessengerEXT VulkanContext::GetDebugMessengerHandle() const {
    return mPDebugUtilsMessenger->GetHandle();
}
#endif

vk::SurfaceKHR VulkanContext::GetSurfaceHandle() const {
    return mPSurface->GetHandle();
}

vk::PhysicalDevice VulkanContext::GetPhysicalDeviceHandle() const {
    return mPPhysicalDevice->GetHandle();
}

vk::Device VulkanContext::GetDeviceHandle() const {
    return mPDevice->GetHandle();
}

VmaAllocator VulkanContext::GetVmaAllocatorHandle() const {
    return mPAllocator->GetHandle();
}

vk::Semaphore VulkanContext::GetTimelineSemaphoreHandle() const {
    return mPTimelineSemaphore->GetHandle();
}

#ifdef CUDA_VULKAN_INTEROP
VmaPool VulkanContext::GetExternalMemoryPoolHandle() const {
    return mPExternalMemoryPool->GetHandle();
}
#endif

vk::Sampler VulkanContext::GetDefaultNearestSamplerHandle() const {
    return mDefaultSamplerNearest->GetHandle();
}

vk::Sampler VulkanContext::GetDefaultLinearSamplerHandle() const {
    return mDefaultSamplerLinear->GetHandle();
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