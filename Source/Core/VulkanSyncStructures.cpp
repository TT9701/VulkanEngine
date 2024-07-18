#include "VulkanSyncStructures.hpp"

#include "VulkanDevice.hpp"

VulkanFence::VulkanFence(Type_SPInstance<VulkanDevice> const& device,
                         vk::FenceCreateFlags flags)
    : pDeivce(device), mFence(CreateFence(flags)) {}

VulkanFence::~VulkanFence() {
    pDeivce->GetHandle().destroy(mFence);
}

vk::Fence VulkanFence::CreateFence(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo fenceCreateInfo {flags};

    return pDeivce->GetHandle().createFence(fenceCreateInfo);
}

VulkanSemaphore::VulkanSemaphore(Type_SPInstance<VulkanDevice> const& device)
    : pDevice(device), mSemaphore(CreateSem()) {}

VulkanSemaphore::~VulkanSemaphore() {
    pDevice->GetHandle().destroy(mSemaphore);
}

vk::Semaphore VulkanSemaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pDevice->GetHandle().createSemaphore(semaphoreCreateInfo);
}