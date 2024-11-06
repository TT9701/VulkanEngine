#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class Fence {
public:
    Fence(Context* ctx,
          vk::FenceCreateFlags flags = vk::FenceCreateFlagBits::eSignaled);
    ~Fence();
    MOVABLE_ONLY(Fence);

public:
    vk::Fence GetHandle() const { return mFence; }

    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

private:
    vk::Fence CreateFence(vk::FenceCreateFlags flags);

private:
    Context* pContext;

    vk::Fence mFence;
};

class Semaphore {
public:
    Semaphore(Context* ctx);
    ~Semaphore();
    MOVABLE_ONLY(Semaphore);

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
    MOVABLE_ONLY(TimelineSemaphore);

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