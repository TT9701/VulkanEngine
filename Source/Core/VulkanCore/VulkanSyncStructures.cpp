#include "VulkanSyncStructures.hpp"

#include "VulkanContext.hpp"

VulkanFence::VulkanFence(VulkanContext* ctx, vk::FenceCreateFlags flags)
    : pContext(ctx), mFence(CreateFence(flags)) {}

VulkanFence::~VulkanFence() {
    pContext->GetDeviceHandle().destroy(mFence);
}

vk::Fence VulkanFence::CreateFence(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo fenceCreateInfo {flags};

    return pContext->GetDeviceHandle().createFence(fenceCreateInfo);
}

VulkanSemaphore::VulkanSemaphore(VulkanContext* ctx)
    : pContext(ctx), mSemaphore(CreateSem()) {}

VulkanSemaphore::~VulkanSemaphore() {
    pContext->GetDeviceHandle().destroy(mSemaphore);
}

vk::Semaphore VulkanSemaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pContext->GetDeviceHandle().createSemaphore(semaphoreCreateInfo);
}

VulkanTimelineSemaphore::VulkanTimelineSemaphore(VulkanContext* ctx,
                                                 uint64_t       initialValue)
    : pContext(ctx),
      mValue(initialValue),
      mSemaphore(CreateTimelineSemaphore()) {}

VulkanTimelineSemaphore::~VulkanTimelineSemaphore() {
    pContext->GetDeviceHandle().destroy(mSemaphore);
}

void VulkanTimelineSemaphore::IncreaseValue(uint64_t val) {
    mValue += val;
}

vk::Semaphore VulkanTimelineSemaphore::CreateTimelineSemaphore() {
    vk::SemaphoreTypeCreateInfoKHR typeInfo {};
    typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
    typeInfo.setInitialValue(mValue);

    vk::SemaphoreCreateInfo info {};
    info.setPNext(&typeInfo);
    return pContext->GetDeviceHandle().createSemaphore(info);
}