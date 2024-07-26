#include "VulkanSyncStructures.hpp"

#include "VulkanContext.hpp"

VulkanFence::VulkanFence(Type_SPInstance<VulkanContext> const& ctx,
                         vk::FenceCreateFlags flags)
    : pContext(ctx), mFence(CreateFence(flags)) {}

VulkanFence::~VulkanFence() {
    pContext->GetDevice()->GetHandle().destroy(mFence);
}

vk::Fence VulkanFence::CreateFence(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo fenceCreateInfo {flags};

    return pContext->GetDevice()->GetHandle().createFence(fenceCreateInfo);
}

VulkanSemaphore::VulkanSemaphore(Type_SPInstance<VulkanContext> const& ctx)
    : pContext(ctx), mSemaphore(CreateSem()) {}

VulkanSemaphore::~VulkanSemaphore() {
    pContext->GetDevice()->GetHandle().destroy(mSemaphore);
}

vk::Semaphore VulkanSemaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pContext->GetDevice()->GetHandle().createSemaphore(
        semaphoreCreateInfo);
}