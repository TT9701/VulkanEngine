#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanHelper.hpp"

class VulkanDevice;

class VulkanFence {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanFence(Type_SPInstance<VulkanDevice> const& device,
                vk::FenceCreateFlags flags);
    ~VulkanFence();

    vk::Fence GetHandle() const { return mFence; }

private:
    vk::Fence CreateFence(vk::FenceCreateFlags flags);

private:
    Type_SPInstance<VulkanDevice> pDeivce;

    vk::Fence mFence;
};

class VulkanSemaphore {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanSemaphore(Type_SPInstance<VulkanDevice> const& device);
    ~VulkanSemaphore();

    vk::Semaphore GetHandle() const { return mSemaphore; }

private:
    vk::Semaphore CreateSem();

private:
    Type_SPInstance<VulkanDevice> pDevice;

    vk::Semaphore mSemaphore;
};