#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Utilities/MemoryPool.hpp"

class VulkanContext;

class VulkanFence {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanFence(
        Type_SPInstance<VulkanContext> const& ctx,
        vk::FenceCreateFlags flags = vk::FenceCreateFlagBits::eSignaled);
    ~VulkanFence();
    MOVABLE_ONLY(VulkanFence);

public:
    vk::Fence const& GetHandle() const { return mFence; }

    static constexpr uint64_t TIME_OUT_NANO_SECONDS = 1000000000;

private:
    vk::Fence CreateFence(vk::FenceCreateFlags flags);

private:
    Type_SPInstance<VulkanContext> pContext;

    vk::Fence mFence;
};

class VulkanSemaphore {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanSemaphore(Type_SPInstance<VulkanContext> const& ctx);
    ~VulkanSemaphore();
    MOVABLE_ONLY(VulkanSemaphore);

public:
    vk::Semaphore const& GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    Type_SPInstance<VulkanContext> pContext;

    vk::Semaphore mSemaphore;
};