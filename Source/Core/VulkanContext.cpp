#include "VulkanContext.hpp"

#include "Core/Utilities/MemoryPool.hpp"
#include "VulkanDebugUtils.hpp"
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"
#include "VulkanPhysicalDevice.hpp"
#include "VulkanSurface.hpp"

vk::PhysicalDeviceFeatures VulkanContext::sPhysicalDeviceFeatures {};
vk::PhysicalDeviceVulkan11Features VulkanContext::sEnable11Features {};
vk::PhysicalDeviceVulkan12Features VulkanContext::sEnable12Features {};
vk::PhysicalDeviceVulkan13Features VulkanContext::sEnable13Features {};

VulkanContext::VulkanContext(
    Type_SPInstance<SDLWindow> const& window,
    vk::QueueFlags requestedQueueFlags,
    ::std::vector<::std::string> const& requestedInstanceLayers,
    ::std::vector<::std::string> const& requestedInstanceExtensions,
    ::std::vector<::std::string> const& requestedDeviceExtensions)
    : mSPInstance(
          CreateInstance(requestedInstanceLayers, requestedInstanceExtensions)),
#ifdef DEBUG
      mSPDebugUtilsMessenger(CreateDebugUtilsMessenger()),
#endif
      mSPSurface(CreateSurface(window)),
      mSPPhysicalDevice(PickPhysicalDevice(requestedQueueFlags)),
      mSPDevice(CreateDevice(requestedDeviceExtensions)) {
}

vk::Instance const& VulkanContext::GetInstanceHandle() const {
    return mSPInstance->GetHandle();
}

vk::DebugUtilsMessengerEXT const& VulkanContext::GetDebugMessengerHandle()
    const {
    return mSPDebugUtilsMessenger->GetHandle();
}

VkSurfaceKHR const& VulkanContext::GetSurfaceHandle() const {
    return mSPSurface->GetHandle();
}

vk::PhysicalDevice const& VulkanContext::GetPhysicalDeviceHandle() const {
    return mSPPhysicalDevice->GetHandle();
}

vk::Device const& VulkanContext::GetDeviceHandle() const {
    return mSPDevice->GetHandle();
}

VulkanContext::Type_SPInstance<VulkanInstance> VulkanContext::CreateInstance(
    ::std::vector<::std::string> const& requestedLayers,
    ::std::vector<::std::string> const& requestedExtensions) {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanInstance>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), requestedLayers,
        requestedExtensions);
}

#ifdef DEBUG
VulkanContext::Type_SPInstance<VulkanDebugUtils>
VulkanContext::CreateDebugUtilsMessenger() {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanDebugUtils>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), mSPInstance);
}
#endif

VulkanContext::Type_SPInstance<VulkanSurface> VulkanContext::CreateSurface(
    Type_SPInstance<SDLWindow> const& window) {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanSurface>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), mSPInstance, window);
}

VulkanContext::Type_SPInstance<VulkanPhysicalDevice>
VulkanContext::PickPhysicalDevice(vk::QueueFlags flags) {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanPhysicalDevice>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), mSPInstance, flags);
}

VulkanContext::Type_SPInstance<VulkanDevice> VulkanContext::CreateDevice(
    ::std::vector<::std::string> const& requestedExtensions) {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<VulkanDevice>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), mSPPhysicalDevice,
        ::std::vector<::std::string> {}, requestedExtensions,
        &sPhysicalDeviceFeatures, &sEnable11Features);
}

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