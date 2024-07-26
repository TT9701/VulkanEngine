#include "VulkanSyncStructures.hpp"

#include "VulkanContext.hpp"

VulkanFence::VulkanFence(Type_SPInstance<VulkanContext> const& ctx,
                         vk::FenceCreateFlags flags)
    : pContext(ctx), mFence(CreateFence(flags)) {}

VulkanFence::~VulkanFence() {
    pContext->GetDeviceHandle().destroy(mFence);
}

vk::Fence VulkanFence::CreateFence(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo fenceCreateInfo {flags};

    return pContext->GetDeviceHandle().createFence(fenceCreateInfo);
}

VulkanSemaphore::VulkanSemaphore(Type_SPInstance<VulkanContext> const& ctx)
    : pContext(ctx), mSemaphore(CreateSem()) {}

VulkanSemaphore::~VulkanSemaphore() {
    pContext->GetDeviceHandle().destroy(mSemaphore);
}

vk::Semaphore VulkanSemaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pContext->GetDeviceHandle().createSemaphore(semaphoreCreateInfo);
}