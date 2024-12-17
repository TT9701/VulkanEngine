#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

class FencePool {
public:
    FencePool(VulkanContext& ctx);
    ~FencePool();

public:
    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

    vk::Fence RequestFence(vk::FenceCreateFlags flags = {});

    vk::Result Wait() const;

    vk::Result Reset();

private:
    VulkanContext& mContext;

    Type_STLVector<vk::Fence> mFences;
    uint32_t mActiveFenceCount {0};
};

class Semaphore {
public:
    Semaphore(VulkanContext& ctx);
    ~Semaphore();
    CLASS_MOVABLE_ONLY(Semaphore);

public:
    vk::Semaphore GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    VulkanContext& mContext;

    vk::Semaphore mSemaphore;
};

class SemaphorePool {
public:
    SemaphorePool(VulkanContext& ctx);
    ~SemaphorePool();

    CLASS_NO_COPY_MOVE(SemaphorePool);

    vk::Semaphore RequestSemaphore();

    vk::Semaphore RequestSemaphore_WithOwnership();

    void ReleaseOwnedSemaphore(vk::Semaphore semaphore);

    void Reset();

    uint32_t GetActiveSemaphoreCount() const;

private:
    VulkanContext& mContext;

    Type_STLVector<vk::Semaphore> mSemaphores;
    Type_STLVector<vk::Semaphore> mReleasedSemaphores;

    uint32_t mActiveSemaphoreCount {0};
};

class TimelineSemaphore {
public:
    TimelineSemaphore(VulkanContext& ctx, uint64_t initialValue = 0ui64);
    ~TimelineSemaphore();
    CLASS_MOVABLE_ONLY(TimelineSemaphore);

public:
    vk::Semaphore GetHandle() const { return mSemaphore; }

    uint64_t GetValue() const { return mValue; }

    uint64_t const* GetValueAddress() const { return &mValue; }

    void IncreaseValue(uint64_t val = 1);

private:
    vk::Semaphore CreateTimelineSemaphore();

private:
    VulkanContext& mContext;

    uint64_t mValue {0};
    vk::Semaphore mSemaphore;
};

}  // namespace IntelliDesign_NS::Vulkan::Core