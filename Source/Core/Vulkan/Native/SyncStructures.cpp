#include "SyncStructures.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

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

Semaphore::Semaphore(VulkanContext& ctx)
    : mContext(ctx), mSemaphore(CreateSem()) {}

Semaphore::~Semaphore() {
    mContext.GetDevice()->destroy(mSemaphore);
}

vk::Semaphore Semaphore::CreateSem() {
    vk::SemaphoreCreateInfo semaphoreCreateInfo {};

    return mContext.GetDevice()->createSemaphore(semaphoreCreateInfo);
}

SemaphorePool::SemaphorePool(VulkanContext& ctx) : mContext(ctx) {}

SemaphorePool::~SemaphorePool() {
    Reset();

    for (auto sem : mSemaphores) {
        mContext.GetDevice()->destroy(sem);
    }

    mSemaphores.clear();
}

vk::Semaphore SemaphorePool::RequestSemaphore() {
    if (mActiveSemaphoreCount < mSemaphores.size()) {
        return mSemaphores[mActiveSemaphoreCount++];
    }

    auto semaphore =
        mContext.GetDevice()->createSemaphore(vk::SemaphoreCreateInfo {});

    mSemaphores.push_back(semaphore);

    mActiveSemaphoreCount++;

    return semaphore;
}

vk::Semaphore SemaphorePool::RequestSemaphore_WithOwnership() {
    if (mActiveSemaphoreCount < mSemaphores.size()) {
        auto semaphore = mSemaphores.back();
        mSemaphores.pop_back();
        return semaphore;
    }

    auto semaphore =
        mContext.GetDevice()->createSemaphore(vk::SemaphoreCreateInfo {});

    return semaphore;
}

void SemaphorePool::ReleaseOwnedSemaphore(vk::Semaphore semaphore) {
    // We cannot reuse this semaphore until ::reset().
    mReleasedSemaphores.push_back(semaphore);
}

void SemaphorePool::Reset() {
    mActiveSemaphoreCount = 0;

    // Now we can safely recycle the released semaphores.
    for (auto& sem : mReleasedSemaphores) {
        mSemaphores.push_back(sem);
    }

    mReleasedSemaphores.clear();
}

uint32_t SemaphorePool::GetActiveSemaphoreCount() const {
    return mActiveSemaphoreCount;
}

TimelineSemaphore::TimelineSemaphore(VulkanContext& ctx, uint64_t initialValue)
    : mContext(ctx),
      mValue(initialValue),
      mSemaphore(CreateTimelineSemaphore()) {}

TimelineSemaphore::~TimelineSemaphore() {
    mContext.GetDevice()->destroy(mSemaphore);
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
    return mContext.GetDevice()->createSemaphore(info);
}

}  // namespace IntelliDesign_NS::Vulkan::Core