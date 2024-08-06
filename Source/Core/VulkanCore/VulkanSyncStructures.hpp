#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanFence {
public:
    VulkanFence(VulkanContext* ctx, vk::FenceCreateFlags flags =
                                        vk::FenceCreateFlagBits::eSignaled);
    ~VulkanFence();
    MOVABLE_ONLY(VulkanFence);

public:
    vk::Fence GetHandle() const { return mFence; }

    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

private:
    vk::Fence CreateFence(vk::FenceCreateFlags flags);

private:
    VulkanContext* pContext;

    vk::Fence mFence;
};

class VulkanSemaphore {
public:
    VulkanSemaphore(VulkanContext* ctx);
    ~VulkanSemaphore();
    MOVABLE_ONLY(VulkanSemaphore);

public:
    vk::Semaphore GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    VulkanContext* pContext;

    vk::Semaphore mSemaphore;
};

class VulkanTimelineSemaphore {
public:
    VulkanTimelineSemaphore(VulkanContext* ctx, uint64_t initialValue = 0ui64);
    ~VulkanTimelineSemaphore();
    MOVABLE_ONLY(VulkanTimelineSemaphore);

public:
    vk::Semaphore GetHandle() const { return mSemaphore; }

    uint64_t GetValue() const { return mValue; }

    uint64_t const* GetValueAddress() const { return &mValue; }

    void IncreaseValue(uint64_t val = 1);

private:
    vk::Semaphore CreateTimelineSemaphore();

private:
    VulkanContext* pContext;

    uint64_t mValue {0};
    vk::Semaphore mSemaphore;
};