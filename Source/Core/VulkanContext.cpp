#include "VulkanContext.hpp"

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
    mDefaultSamplerNearest = MakeUnique<VulkanSampler>(
        this, vk::Filter::eNearest, vk::Filter::eNearest);
    mDefaultSamplerLinear = MakeUnique<VulkanSampler>(this, vk::Filter::eLinear,
                                                      vk::Filter::eLinear);
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