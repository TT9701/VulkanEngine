#include "Context.hpp"

#include "CUDA/CUDAVulkan.h"

namespace IntelliDesign_NS::Vulkan::Core {

vk::PhysicalDeviceFeatures Context::sPhysicalDeviceFeatures {};
vk::PhysicalDeviceVulkan11Features Context::sEnable11Features {};
vk::PhysicalDeviceVulkan12Features Context::sEnable12Features {};
vk::PhysicalDeviceVulkan13Features Context::sEnable13Features {};

Context::Context(const SDLWindow* window, vk::QueueFlags requestedQueueFlags,
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

    mPDevice->SetObjectName(mPInstance->GetHandle(), "Default Instance");
    mPDevice->SetObjectName(mPSurface->GetHandle(), "Default Surface");
    mPDevice->SetObjectName(
        mPPhysicalDevice->GetHandle(),
        mPPhysicalDevice->GetHandle().getProperties2().properties.deviceName);
    mPDevice->SetObjectName(mPDevice->GetHandle(), "Default Device");
    mPDevice->SetObjectName(mPTimelineSemaphore->GetHandle(),
                            "Main Timeline Semaphore");

    mPDevice->SetObjectName(mDefaultSamplerLinear->GetHandle(),
                            "Default Linear Sampler");
    mPDevice->SetObjectName(mDefaultSamplerNearest->GetHandle(),
                            "Default Nearest Sampler");

    mPDevice->SetObjectName(mPTimelineSemaphore->GetHandle(),
                            "Main Timeline Semaphore");
}

SharedPtr<RenderResource> Context::CreateTexture2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    uint32_t mipLevels, uint32_t arraySize, uint32_t sampleCount) {
    return MakeShared<RenderResource>(
        mPDevice.get(), mPAllocator.get(), RenderResource::Type::Texture2D,
        format, extent, usage, mipLevels, arraySize, sampleCount);
}

SharedPtr<RenderResource> Context::CreateDeviceLocalBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<RenderResource>(
        mPDevice.get(), mPAllocator.get(), RenderResource::Type::Buffer,
        allocByteSize, usage, Buffer::MemoryType::DeviceLocal);
}

SharedPtr<RenderResource> Context::CreateStagingBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<RenderResource>(
        mPDevice.get(), mPAllocator.get(), RenderResource::Type::Buffer,
        allocByteSize, usage | vk::BufferUsageFlagBits::eTransferSrc,
        Buffer::MemoryType::Staging);
}

SharedPtr<RenderResource> Context::CreateStorageBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return CreateDeviceLocalBuffer(
        allocByteSize, usage | vk::BufferUsageFlagBits::eStorageBuffer);
}

SharedPtr<RenderResource> Context::CreateIndirectCmdBuffer(
    size_t allocByteSize) {
    return CreateDeviceLocalBuffer(allocByteSize,
                                   vk::BufferUsageFlagBits::eIndirectBuffer
                                       | vk::BufferUsageFlagBits::eTransferDst);
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> Context::CreateExternalImage2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags,
    uint32_t mipmapLevels, uint32_t arrayLayers) {
    return MakeShared<CUDA::VulkanExternalImage>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(),
        mPExternalMemoryPool->GetHandle(), flags, extent, format, usage, aspect,
        mipmapLevels, arrayLayers, vk::ImageType::e2D, vk::ImageViewType::e2D);
}

SharedPtr<CUDA::VulkanExternalBuffer> Context::CreateExternalPersistentBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mPExternalMemoryPool->GetHandle());
}

SharedPtr<CUDA::VulkanExternalBuffer> Context::CreateExternalStagingBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mPDevice->GetHandle(), mPAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        mPExternalMemoryPool->GetHandle());
}
#endif

SharedPtr<Sampler> Context::CreateSampler(vk::Filter minFilter,
                                          vk::Filter magFilter,
                                          vk::SamplerAddressMode addressModeU,
                                          vk::SamplerAddressMode addressModeV,
                                          vk::SamplerAddressMode addressModeW,
                                          float maxLod, bool compareEnable,
                                          vk::CompareOp compareOp) {
    return MakeShared<Sampler>(this, minFilter, magFilter, addressModeU,
                               addressModeV, addressModeW, maxLod,
                               compareEnable, compareOp);
}

Instance* Context::GetInstance() const {
    return mPInstance.get();
}

#ifndef NDEBUG
DebugUtils* Context::GetDebugMessenger() const {
    return mPDebugUtilsMessenger.get();
}
#endif

Surface* Context::GetSurface() const {
    return mPSurface.get();
}

PhysicalDevice* Context::GetPhysicalDevice() const {
    return mPPhysicalDevice.get();
}

Device* Context::GetDevice() const {
    return mPDevice.get();
}

MemoryAllocator* Context::GetVmaAllocator() const {
    return mPAllocator.get();
}

TimelineSemaphore* Context::GetTimelineSemphore() const {
    return mPTimelineSemaphore.get();
}

#ifdef CUDA_VULKAN_INTEROP
ExternalMemoryPool* Context::GetExternalMemoryPool() const {
    return mPExternalMemoryPool.get();
}
#endif

Sampler* Context::GetDefaultNearestSampler() const {
    return mDefaultSamplerNearest.get();
}

Sampler* Context::GetDefaultLinearSampler() const {
    return mDefaultSamplerLinear.get();
}

vk::Instance Context::GetInstanceHandle() const {
    return mPInstance->GetHandle();
}

#ifndef NDEBUG
vk::DebugUtilsMessengerEXT Context::GetDebugMessengerHandle() const {
    return mPDebugUtilsMessenger->GetHandle();
}
#endif

vk::SurfaceKHR Context::GetSurfaceHandle() const {
    return mPSurface->GetHandle();
}

vk::PhysicalDevice Context::GetPhysicalDeviceHandle() const {
    return mPPhysicalDevice->GetHandle();
}

vk::Device Context::GetDeviceHandle() const {
    return mPDevice->GetHandle();
}

VmaAllocator Context::GetVmaAllocatorHandle() const {
    return mPAllocator->GetHandle();
}

vk::Semaphore Context::GetTimelineSemaphoreHandle() const {
    return mPTimelineSemaphore->GetHandle();
}

#ifdef CUDA_VULKAN_INTEROP
VmaPool Context::GetExternalMemoryPoolHandle() const {
    return mPExternalMemoryPool->GetHandle();
}
#endif

vk::Sampler Context::GetDefaultNearestSamplerHandle() const {
    return mDefaultSamplerNearest->GetHandle();
}

vk::Sampler Context::GetDefaultLinearSamplerHandle() const {
    return mDefaultSamplerLinear->GetHandle();
}

UniquePtr<Instance> Context::CreateInstance(
    ::std::span<::std::string> requestedLayers,
    ::std::span<::std::string> requestedExtensions) {
    return MakeUnique<Instance>(requestedLayers, requestedExtensions);
}

#ifndef NDEBUG
UniquePtr<DebugUtils> Context::CreateDebugUtilsMessenger() {
    return MakeUnique<DebugUtils>(mPInstance.get());
}
#endif

UniquePtr<Surface> Context::CreateSurface(const SDLWindow* window) {
    return MakeUnique<Surface>(mPInstance.get(), window);
}

UniquePtr<PhysicalDevice> Context::PickPhysicalDevice(vk::QueueFlags flags) {
    return MakeUnique<PhysicalDevice>(mPInstance.get(), flags);
}

UniquePtr<Device> Context::CreateDevice(
    ::std::span<::std::string> requestedExtensions) {
    return MakeUnique<Device>(
        mPPhysicalDevice.get(), ::std::span<::std::string> {},
        requestedExtensions, &sPhysicalDeviceFeatures, &sEnable11Features);
}

UniquePtr<MemoryAllocator> Context::CreateVmaAllocator() {
    return MakeUnique<MemoryAllocator>(mPPhysicalDevice.get(), mPDevice.get(),
                                       mPInstance.get());
}

UniquePtr<TimelineSemaphore> Context::CreateTimelineSem() {
    return MakeUnique<TimelineSemaphore>(this);
}

#ifdef CUDA_VULKAN_INTEROP
UniquePtr<ExternalMemoryPool> Context::CreateExternalMemoryPool() {
    return MakeUnique<ExternalMemoryPool>(mPAllocator.get());
}

void Context::CreateDefaultSamplers() {
    mDefaultSamplerNearest =
        CreateSampler(vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear =
        CreateSampler(vk::Filter::eLinear, vk::Filter::eLinear);
}
#endif

void Context::EnableDefaultFeatures() {
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

void Context::EnableDynamicRendering() {
    sEnable13Features.setDynamicRendering(vk::True);
}

void Context::EnableSynchronization2() {
    sEnable13Features.setSynchronization2(vk::True);
}

void Context::EnableBufferDeviceAddress() {
    sEnable12Features.setBufferDeviceAddress(vk::True);
}

void Context::EnableDescriptorIndexing() {
    sEnable12Features.setDescriptorIndexing(vk::True);
}

void Context::EnableTimelineSemaphore() {
    sEnable12Features.setTimelineSemaphore(vk::True);
}

void Context::EnableMultiDrawIndirect() {
    sPhysicalDeviceFeatures.setMultiDrawIndirect(vk::True);
    sPhysicalDeviceFeatures.setDrawIndirectFirstInstance(vk::True);
}

}  // namespace IntelliDesign_NS::Vulkan::Core