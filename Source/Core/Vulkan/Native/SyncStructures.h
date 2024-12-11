#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class FencePool {
public:
    FencePool(Context* ctx);
    ~FencePool();

public:
    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

    vk::Fence RequestFence(
        vk::FenceCreateFlags flags = {});

    vk::Result Wait() const;

    vk::Result Reset();

private:
    Context* pContext;

    Type_STLVector<vk::Fence> mFences;
    uint32_t mActiveFenceCount {0};
};

class Semaphore {
public:
    Semaphore(Context* ctx);
    ~Semaphore();
    CLASS_MOVABLE_ONLY(Semaphore);

public:
    vk::Semaphore GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    Context* pContext;

    vk::Semaphore mSemaphore;
};

class TimelineSemaphore {
public:
    TimelineSemaphore(Context* ctx, uint64_t initialValue = 0ui64);
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
    Context* pContext;

    uint64_t mValue {0};
    vk::Semaphore mSemaphore;
};

}  // namespace IntelliDesign_NS::Vulkan::Core