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