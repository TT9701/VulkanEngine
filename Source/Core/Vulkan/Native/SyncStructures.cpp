#include "SyncStructures.h"

#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

FencePool::FencePool(Context* ctx) : pContext(ctx) {}

FencePool::~FencePool() {
    Wait();
    Reset();

    for (auto& fence : mFences) {
        pContext->GetDeviceHandle().destroy(fence);
    }

    mFences.clear();
}

vk::Fence FencePool::RequestFence(vk::FenceCreateFlags flags) {
    if (mActiveFenceCount < mFences.size()) {
        return mFences[mActiveFenceCount++];
    }

    vk::FenceCreateInfo info {flags};

    auto fence = pContext->GetDeviceHandle().createFence(info);

    mFences.push_back(fence);

    mActiveFenceCount++;

    return mFences.back();
}

vk::Result FencePool::Wait() const {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }

    return pContext->GetDeviceHandle().waitForFences(
        mFences, vk::True, TIME_OUT_NANO_SECONDS);
}

vk::Result FencePool::Reset() {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }

    pContext->GetDeviceHandle().resetFences(mFences);

    mActiveFenceCount = 0;

    return vk::Result::eSuccess;
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