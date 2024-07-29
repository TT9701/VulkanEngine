#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Utilities/MemoryPool.hpp"

class VulkanContext;

class VulkanFence {
public:
    VulkanFence(
        SharedPtr<VulkanContext> const& ctx,
        vk::FenceCreateFlags flags = vk::FenceCreateFlagBits::eSignaled);
    ~VulkanFence();
    MOVABLE_ONLY(VulkanFence);

public:
    vk::Fence const& GetHandle() const { return mFence; }

    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

private:
    vk::Fence CreateFence(vk::FenceCreateFlags flags);

private:
    SharedPtr<VulkanContext> pContext;

    vk::Fence mFence;
};

class VulkanSemaphore {
public:
    VulkanSemaphore(SharedPtr<VulkanContext> const& ctx);
    ~VulkanSemaphore();
    MOVABLE_ONLY(VulkanSemaphore);

public:
    vk::Semaphore const& GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    SharedPtr<VulkanContext> pContext;

    vk::Semaphore mSemaphore;
};