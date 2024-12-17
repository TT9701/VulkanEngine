#include "VulkanContext.h"

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAVulkan.h"
#endif

namespace IntelliDesign_NS::Vulkan::Core {

VulkanContext::VulkanContext(
    SDLWindow& window, ::std::span<Type_STLString> requestedInstanceLayers,
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

    mFencePool = MakeUnique<FencePool>(*this);
    mCommandPool = MakeUnique<CommandPool>(
        *this, GetQueue(QueueUsage::Graphics).GetFamilyIndex());
}

SharedPtr<Texture> VulkanContext::CreateTexture2D(
    const char* name, vk::Extent3D extent, vk::Format format,
    vk::ImageUsageFlags usage, uint32_t mipLevels, uint32_t arraySize,
    uint32_t sampleCount) {
    auto ptr =
        MakeShared<Texture>(*this, Texture::Type::Texture2D, format, extent,
                            usage, mipLevels, arraySize, sampleCount);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<RenderResource> VulkanContext::CreateDeviceLocalBufferResource(
    const char* name, size_t allocByteSize, vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<RenderResource>(*this, RenderResource::Type::Buffer,
                                          allocByteSize, usage,
                                          Buffer::MemoryType::DeviceLocal);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> VulkanContext::CreateDeviceLocalBuffer(
    const char* name, size_t allocByteSize, vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<Buffer>(*this, allocByteSize, usage,
                                  Buffer::MemoryType::DeviceLocal);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> VulkanContext::CreateStagingBuffer(
    const char* name, size_t allocByteSize, vk::BufferUsageFlags usage) {
    auto ptr = MakeShared<Buffer>(*this, allocByteSize,
                                  usage | vk::BufferUsageFlagBits::eTransferSrc,
                                  Buffer::MemoryType::Staging);
    ptr->SetName(name);
    return ptr;
}

SharedPtr<Buffer> VulkanContext::CreateStorageBuffer(
    const char* name, size_t allocByteSize, vk::BufferUsageFlags usage) {
    return CreateDeviceLocalBuffer(
        name, allocByteSize, usage | vk::BufferUsageFlagBits::eStorageBuffer);
}

SharedPtr<Buffer> VulkanContext::CreateIndirectCmdBuffer(const char* name,
                                                         size_t allocByteSize) {
    return CreateDeviceLocalBuffer(name, allocByteSize,
                                   vk::BufferUsageFlagBits::eIndirectBuffer
                                       | vk::BufferUsageFlagBits::eTransferDst);
}

#ifdef CUDA_VULKAN_INTEROP
SharedPtr<CUDA::VulkanExternalImage> VulkanContext::CreateExternalImage2D(
    vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect, VmaAllocationCreateFlags flags,
    uint32_t mipmapLevels, uint32_t arrayLayers) {
    return MakeShared<CUDA::VulkanExternalImage>(
        mDevice->GetHandle(), mAllocator->GetHandle(),
        mPExternalMemoryPool->GetHandle(), flags, extent, format, usage, aspect,
        mipmapLevels, arrayLayers, vk::ImageType::e2D, vk::ImageViewType::e2D);
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalPersistentBuffer(size_t allocByteSize,
                                              vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mDevice->GetHandle(), mAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        mPExternalMemoryPool->GetHandle());
}

SharedPtr<CUDA::VulkanExternalBuffer>
VulkanContext::CreateExternalStagingBuffer(size_t allocByteSize,
                                           vk::BufferUsageFlags usage) {
    return MakeShared<CUDA::VulkanExternalBuffer>(
        mDevice->GetHandle(), mAllocator->GetHandle(), allocByteSize, usage,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        mPExternalMemoryPool->GetHandle());
}
#endif

SharedPtr<Sampler> VulkanContext::CreateSampler(
    vk::Filter minFilter, vk::Filter magFilter,
    vk::SamplerAddressMode addressModeU, vk::SamplerAddressMode addressModeV,
    vk::SamplerAddressMode addressModeW, float maxLod, bool compareEnable,
    vk::CompareOp compareOp) {
    return MakeShared<Sampler>(this, minFilter, magFilter, addressModeU,
                               addressModeV, addressModeW, maxLod,
                               compareEnable, compareOp);
}

Instance& VulkanContext::GetInstance() const {
    return *mInstance;
}

#ifndef NDEBUG
DebugUtils& VulkanContext::GetDebugMessenger() const {
    return *mDebugUtilsMessenger;
}
#endif

Surface& VulkanContext::GetSurface() const {
    return *mSurface;
}

PhysicalDevice& VulkanContext::GetPhysicalDevice() const {
    return mPhysicalDevice;
}

Device& VulkanContext::GetDevice() const {
    return *mDevice;
}

MemoryAllocator& VulkanContext::GetVmaAllocator() const {
    return *mAllocator;
}

TimelineSemaphore& VulkanContext::GetTimelineSemphore() const {
    return *mTimelineSemaphore;
}

#ifdef CUDA_VULKAN_INTEROP
ExternalMemoryPool* VulkanContext::GetExternalMemoryPool() const {
    return mPExternalMemoryPool.get();
}
#endif

Sampler& VulkanContext::GetDefaultNearestSampler() const {
    return *mDefaultSamplerNearest;
}

Sampler& VulkanContext::GetDefaultLinearSampler() const {
    return *mDefaultSamplerLinear;
}

Queue const& VulkanContext::GetQueue(QueueUsage usage,
                                     bool highPriority) const {
    switch (usage) {
        case QueueUsage::Compute_Runtime:
            return mDevice->GetQueue(
                mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eCompute), 0);
        case QueueUsage::Transfer_Runtime_Upload:
            return mDevice->GetQueue(
                mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer), 0);
        case QueueUsage::Transfer_Runtime_Readback:
            return mDevice->GetQueue(
                mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eTransfer), 1);
        case QueueUsage::Transfer_Runtime_DeviceInternal:
            return mDevice->GetQueue(
                mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics), 2);
        case QueueUsage::Present:
        case QueueUsage::Graphics:
        case QueueUsage::Compute_Prepare:
        case QueueUsage::Transfer_Prepare:
        default:
            if (mPhysicalDevice.HasHighPriorityGraphicsQueue()) {
                if (highPriority) {
                    return mDevice->GetQueue(mDevice->GetQueueFamilyIndex(
                                                 vk::QueueFlagBits::eGraphics),
                                             0);
                } else {
                    return mDevice->GetQueue(mDevice->GetQueueFamilyIndex(
                                                 vk::QueueFlagBits::eGraphics),
                                             1);
                }
            } else {
                return mDevice->GetQueue(
                    mDevice->GetQueueFamilyIndex(vk::QueueFlagBits::eGraphics),
                    0);
            }
    }
}

FencePool& VulkanContext::GetFencePool() const {
    return *mFencePool;
}

CommandPool& VulkanContext::GetCommandPool() const {
    return *mCommandPool;
}

UniquePtr<Instance> VulkanContext::CreateInstance(
    ::std::span<Type_STLString> requestedLayers,
    ::std::span<Type_STLString> requestedExtensions) {
    return MakeUnique<Instance>(requestedLayers, requestedExtensions);
}

#ifndef NDEBUG
UniquePtr<DebugUtils> VulkanContext::CreateDebugUtilsMessenger() {
    return MakeUnique<DebugUtils>(*mInstance);
}
#endif

UniquePtr<Surface> VulkanContext::CreateSurface(SDLWindow& window) {
    return MakeUnique<Surface>(*mInstance, window);
}

UniquePtr<Device> VulkanContext::CreateDevice(
    ::std::span<Type_STLString> requestedExtensions) {
    EnableFeatures();

    return MakeUnique<Device>(mPhysicalDevice, *mSurface, requestedExtensions);
}

UniquePtr<MemoryAllocator> VulkanContext::CreateVmaAllocator() {
    return MakeUnique<MemoryAllocator>(mPhysicalDevice, *mDevice, *mInstance);
}

UniquePtr<TimelineSemaphore> VulkanContext::CreateTimelineSem() {
    return MakeUnique<TimelineSemaphore>(*this);
}

#ifdef CUDA_VULKAN_INTEROP
UniquePtr<ExternalMemoryPool> VulkanContext::CreateExternalMemoryPool() {
    return MakeUnique<ExternalMemoryPool>(mAllocator.get());
}
#endif

void VulkanContext::CreateDefaultSamplers() {
    mDefaultSamplerNearest =
        CreateSampler(vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear =
        CreateSampler(vk::Filter::eLinear, vk::Filter::eLinear);
}

VulkanContext::CmdToBegin::CmdToBegin(Device& device, vk::CommandBuffer cmd,
                                      vk::CommandPool pool, vk::Queue queue,
                                      vk::Semaphore signal)
    : mDevice(device), mHandle(cmd), mPool(pool), mQueue(queue), mSem(signal) {
    cmd.begin(vk::CommandBufferBeginInfo {});
}

VulkanContext::CmdToBegin::~CmdToBegin() {
    mHandle.end();

    vk::SubmitInfo submit_info {{}, {}, mHandle};
    if (mSem) {
        submit_info.setSignalSemaphores(mSem);
    }

    vk::Fence fence = mDevice->createFence({});

    mQueue.submit(submit_info, fence);

    VK_CHECK(
        mDevice->waitForFences(fence, true, FencePool::TIME_OUT_NANO_SECONDS));

    mDevice->destroyFence(fence);

    if (mPool) {
        mDevice->freeCommandBuffers(mPool, mHandle);
    }
}

vk::CommandBuffer const* VulkanContext::CmdToBegin::operator->() const {
    return &mHandle;
}

void VulkanContext::EnableFeatures() {
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

    // REQUEST_REQUIRED_FEATURE(
    //     mPhysicalDevice, vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT,
    //     deviceGeneratedCommands);
}

VulkanContext::CmdToBegin VulkanContext::CreateCmdBufToBegin(
    Queue const& queue, vk::Semaphore signal,
    vk::CommandBufferLevel level) const {
    assert(mCommandPool && "No command pool exists in the device");

    vk::CommandBuffer cmd =
        GetDevice()
            ->allocateCommandBuffers({mCommandPool->GetHandle(), level, 1})
            .front();

    return {*mDevice, cmd, mCommandPool->GetHandle(), queue.GetHandle(),
            signal};
}

}  // namespace IntelliDesign_NS::Vulkan::Core