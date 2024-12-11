#include "Context.h"

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAVulkan.h"
#endif

namespace IntelliDesign_NS::Vulkan::Core {

Context::Context(SDLWindow& window,
                 ::std::span<Type_STLString> requestedInstanceLayers,
                 ::std::span<Type_STLString> requestedInstanceExtensions,
                 ::std::span<Type_STLString> requestedDeviceExtensions)
    : mInstance(
          CreateInstance(requestedInstanceLayers, requestedInstanceExtensions)),
#ifndef NDEBUG
      mDebugUtilsMessenger(CreateDebugUtilsMessenger()),
#endif
      mSurface(CreateSurface(window)),
      mPhysicalDevice(mInstance->GetSuitableGPU(mSurface->GetHandle())),
      mDevice(CreateDevice(requestedDeviceExtensions)),
      mAllocator(CreateVmaAllocator()),
      mTimelineSemaphore(CreateTimelineSem())
#ifdef CUDA_VULKAN_INTEROP
      ,
      mPExternalMemoryPool(CreateExternalMemoryPool())
#endif
{
    CreateDefaultSamplers();

    mDevice->SetObjectName(mInstance->GetHandle(), "Default Instance");
    mDevice->SetObjectName(mSurface->GetHandle(), "Default Surface");
    mDevice->SetObjectName(mPhysicalDevice.GetHandle(),
                           mPhysicalDevice.GetProperties().deviceName);
    mDevice->SetObjectName(mDevice->GetHandle(), "Default Device");
    mDevice->SetObjectName(mTimelineSemaphore->GetHandle(),
                           "Main Timeline Semaphore");

    mDevice->SetObjectName(mDefaultSamplerLinear->GetHandle(),
                           "Default Linear Sampler");
    mDevice->SetObjectName(mDefaultSamplerNearest->GetHandle(),
                           "Default Nearest Sampler");

    mDevice->SetObjectName(mTimelineSemaphore->GetHandle(),
                           "Main Timeline Semaphore");
}

SharedPtr<Texture> Context::CreateTexture2D(
    const char* name, vk::Extent3D extent, vk::Format format,
    vk::ImageUsageFlags usage, uint32_t mipLevels, uint32_t arraySize,
    uint32_t sampleCount) {
    auto ptr = MakeShared<Texture>(mDevice.get(), mAllocator.get(),
                                   Texture::Type::Texture2D, format, extent,
                                   usage, mipLevels, arraySize, sampleCount);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<RenderResource> Context::CreateDeviceLocalBufferResource(
    const char* name, size_t allocByteSize, vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<RenderResource>(
        mDevice.get(), *mAllocator, RenderResource::Type::Buffer, allocByteSize,
        usage, Buffer::MemoryType::DeviceLocal);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> Context::CreateDeviceLocalBuffer(const char* name,
                                                   size_t allocByteSize,
                                                   vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<Buffer>(mDevice.get(), *mAllocator, allocByteSize,
                                  usage, Buffer::MemoryType::DeviceLocal);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> Context::CreateStagingBuffer(const char* name,
                                               size_t allocByteSize,
                                               vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<Buffer>(mDevice.get(), *mAllocator, allocByteSize,
                                  usage | vk::BufferUsageFlagBits::eTransferSrc,
                                  Buffer::MemoryType::Staging);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> Context::CreateStorageBuffer(const char* name,
                                               size_t allocByteSize,
                                               vk::BufferUsageFlags usage) {
    return CreateDeviceLocalBuffer(
        name, allocByteSize, usage | vk::BufferUsageFlagBits::eStorageBuffer);
}

SharedPtr<Buffer> Context::CreateIndirectCmdBuffer(const char* name,
                                                   size_t allocByteSize) {
    return CreateDeviceLocalBuffer(name, allocByteSize,
                                   vk::BufferUsageFlagBits::eIndirectBuffer
                                       | vk::BufferUsageFlagBits::eTransferDst);
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> Context::CreateExternalImage2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags,
    uint32_t mipmapLevels, uint32_t arrayLayers) {
    return MakeShared<CUDA::VulkanExternalImage>(
        mDevice->GetHandle(), mAllocator->GetHandle(),
        mPExternalMemoryPool->GetHandle(), flags, extent, format, usage, aspect,
        mipmapLevels, arrayLayers, vk::ImageType::e2D, vk::ImageViewType::e2D);
}

SharedPtr<CUDA::VulkanExternalBuffer> Context::CreateExternalPersistentBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mDevice->GetHandle(), mAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mPExternalMemoryPool->GetHandle());
}

SharedPtr<CUDA::VulkanExternalBuffer> Context::CreateExternalStagingBuffer(
    size_t allocByteSize, vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mDevice->GetHandle(), mAllocator->GetHandle(), allocByteSize, usage,
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

Instance& Context::GetInstance() const {
    return *mInstance;
}

#ifndef NDEBUG
DebugUtils& Context::GetDebugMessenger() const {
    return *mDebugUtilsMessenger;
}
#endif

Surface& Context::GetSurface() const {
    return *mSurface;
}

PhysicalDevice& Context::GetPhysicalDevice() const {
    return mPhysicalDevice;
}

Device& Context::GetDevice() const {
    return *mDevice;
}

MemoryAllocator& Context::GetVmaAllocator() const {
    return *mAllocator;
}

TimelineSemaphore& Context::GetTimelineSemphore() const {
    return *mTimelineSemaphore;
}

#ifdef CUDA_VULKAN_INTEROP
ExternalMemoryPool* Context::GetExternalMemoryPool() const {
    return mPExternalMemoryPool.get();
}
#endif

Sampler& Context::GetDefaultNearestSampler() const {
    return *mDefaultSamplerNearest;
}

Sampler& Context::GetDefaultLinearSampler() const {
    return *mDefaultSamplerLinear;
}

Queue const& Context::GetPresentQueue() const {
    return GetGraphicsQueue();
}

Queue const& Context::GetGraphicsQueue() const {
    return mDevice->GetQueue(
        mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics), 0);
}

Queue const& Context::GetComputeQueue() const {
    return mDevice->GetQueue(
        mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eCompute), 0);
}

Queue const& Context::GetTransferQueue_ForUpload() const {
    return mDevice->GetQueue(
        mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer), 0);
}

Queue const& Context::GetTransferQueue_ForReadback() const {
    return mDevice->GetQueue(
        mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer), 1);
}

Queue const& Context::GetTransferQueue_ForInternal() const {
    return mDevice->GetQueue(
        mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics), 2);
}

UniquePtr<Instance> Context::CreateInstance(
    ::std::span<Type_STLString> requestedLayers,
    ::std::span<Type_STLString> requestedExtensions) {
    return MakeUnique<Instance>(requestedLayers, requestedExtensions);
}

#ifndef NDEBUG
UniquePtr<DebugUtils> Context::CreateDebugUtilsMessenger() {
    return MakeUnique<DebugUtils>(*mInstance);
}
#endif

UniquePtr<Surface> Context::CreateSurface(SDLWindow& window) {
    return MakeUnique<Surface>(*mInstance, window);
}

UniquePtr<Device> Context::CreateDevice(
    ::std::span<Type_STLString> requestedExtensions) {
    EnableFeatures();

    return MakeUnique<Device>(mPhysicalDevice, *mSurface, requestedExtensions);
}

UniquePtr<MemoryAllocator> Context::CreateVmaAllocator() {
    return MakeUnique<MemoryAllocator>(mPhysicalDevice, *mDevice, *mInstance);
}

UniquePtr<TimelineSemaphore> Context::CreateTimelineSem() {
    return MakeUnique<TimelineSemaphore>(this);
}

#ifdef CUDA_VULKAN_INTEROP
UniquePtr<ExternalMemoryPool> Context::CreateExternalMemoryPool() {
    return MakeUnique<ExternalMemoryPool>(mAllocator.get());
}
#endif

void Context::CreateDefaultSamplers() {
    mDefaultSamplerNearest =
        CreateSampler(vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear =
        CreateSampler(vk::Filter::eLinear, vk::Filter::eLinear);
}

void Context::EnableFeatures() {
    auto& features = mPhysicalDevice.GetMutableRequestedFeatures();
    features.setMultiDrawIndirect(vk::True).setDrawIndirectFirstInstance(
        vk::True);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceShaderDrawParametersFeatures,
                             shaderDrawParameters);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceDynamicRenderingFeatures,
                             dynamicRendering);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceSynchronization2Features,
                             synchronization2);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceBufferDeviceAddressFeatures,
                             bufferDeviceAddress);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceDescriptorIndexingFeatures,
                             descriptorBindingVariableDescriptorCount);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceDescriptorIndexingFeatures,
                             runtimeDescriptorArray);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceDescriptorIndexingFeatures,
                             shaderSampledImageArrayNonUniformIndexing);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceTimelineSemaphoreFeatures,
                             timelineSemaphore);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceMaintenance6FeaturesKHR,
                             maintenance6);

    REQUEST_REQUIRED_FEATURE(
        mPhysicalDevice, vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShader);

    REQUEST_REQUIRED_FEATURE(
        mPhysicalDevice, vk::PhysicalDeviceMeshShaderFeaturesEXT, taskShader);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDeviceDescriptorBufferFeaturesEXT,
                             descriptorBuffer);

    REQUEST_REQUIRED_FEATURE(mPhysicalDevice,
                             vk::PhysicalDevice8BitStorageFeatures,
                             storageBuffer8BitAccess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core