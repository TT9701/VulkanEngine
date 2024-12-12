#include "SyncStructures.h"

#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

FencePool::FencePool(VulkanContext& ctx) : mContext(ctx) {}

FencePool::~FencePool() {
    Wait();
    Reset();

    for (auto& fence : mFences) {
        mContext.GetDevice()->destroy(fence);
    }

    mFences.clear();
}

vk::Fence FencePool::RequestFence(vk::FenceCreateFlags flags) {
    if (mActiveFenceCount < mFences.size()) {
        return mFences[mActiveFenceCount++];
    }

    vk::FenceCreateInfo info {flags};

    auto fence = mContext.GetDevice()->createFence(info);

    mFences.push_back(fence);

    mActiveFenceCount++;

    return mFences.back();
}

vk::Result FencePool::Wait() const {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }

    return mContext.GetDevice()->waitForFences(mFences, vk::True,
                                                TIME_OUT_NANO_SECONDS);
}

vk::Result FencePool::Reset() {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }

    mContext.GetDevice()->resetFences(mFences);

    mActiveFenceCount = 0;

    return vk::Result::eSuccess;
}

Semaphore::Semaphore(VulkanContext* ctx) : pContext(ctx), mSemaphore(CreateSem()) {}

Semaphore::~Semaphore() {
    pContext->GetDevice()->destroy(mSemaphore);
}

vk::Semaphore Semaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return pContext->GetDevice()->createSemaphore(semaphoreCreateInfo);
}

TimelineSemaphore::TimelineSemaphore(VulkanContext* ctx, uint64_t initialValue)
    : pContext(ctx),
      mValue(initialValue),
      mSemaphore(CreateTimelineSemaphore()) {}

TimelineSemaphore::~TimelineSemaphore() {
    pContext->GetDevice()->destroy(mSemaphore);
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
    return pContext->GetDevice()->createSemaphore(info);
}

}  // namespace IntelliDesign_NS::Vulkan::Core