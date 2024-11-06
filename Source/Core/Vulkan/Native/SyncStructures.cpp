#include "SyncStructures.h"

#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

Fence::Fence(Context* ctx, vk::FenceCreateFlags flags)
    : pContext(ctx), mFence(CreateFence(flags)) {}

Fence::~Fence() {
    pContext->GetDeviceHandle().destroy(mFence);
}

vk::Fence Fence::CreateFence(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo fenceCreateInfo {flags};

    return pContext->GetDeviceHandle().createFence(fenceCreateInfo);
}

Semaphore::Semaphore(Context* ctx) : pContext(ctx), mSemaphore(CreateSem()) {}

Semaphore::~Semaphore() {
    pContext->GetDeviceHandle().destroy(mSemaphore);
}

vk::Semaphore Semaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pContext->GetDeviceHandle().createSemaphore(semaphoreCreateInfo);
}

TimelineSemaphore::TimelineSemaphore(Context* ctx, uint64_t initialValue)
    : pContext(ctx),
      mValue(initialValue),
      mSemaphore(CreateTimelineSemaphore()) {}

TimelineSemaphore::~TimelineSemaphore() {
    pContext->GetDeviceHandle().destroy(mSemaphore);
}

void TimelineSemaphore::IncreaseValue(uint64_t val) {
    mValue += val;
}

vk::Semaphore TimelineSemaphore::CreateTimelineSemaphore() {
    vk::SemaphoreTypeCreateInfoKHR typeInfo {};
    typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
    typeInfo.setInitialValue(mValue);

    vk::SemaphoreCreateInfo info {};
    info.setPNext(&typeInfo);
    return pContext->GetDeviceHandle().createSemaphore(info);
}

}  // namespace IntelliDesign_NS::Vulkan::Core