#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanCommandPool {
public:
    VulkanCommandPool(SharedPtr<VulkanContext> ctx,
                      uint32_t queueFamilysIndex,
                      vk::CommandPoolCreateFlags flags =
                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    ~VulkanCommandPool();
    MOVABLE_ONLY(VulkanCommandPool);

public:
    vk::CommandPool GetHandle() const { return mCmdPool; }

private:
    vk::CommandPool CreateCommandPool();

private:
    SharedPtr<VulkanContext> pCtx;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t mQueueFamilysIndex;

    vk::CommandPool mCmdPool;
};

class VulkanCommandBuffer {
public:
    VulkanCommandBuffer(
        SharedPtr<VulkanContext> ctx,
        SharedPtr<VulkanCommandPool> pool,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
        uint32_t count = 1u);
    ~VulkanCommandBuffer() = default;

public:
    vk::CommandBuffer GetHandle(uint32_t index = 0) const {
        return mCmdBuffer[index];
    }

private:
    ::std::vector<vk::CommandBuffer> CreateCommandBuffer(uint32_t count);

private:
    SharedPtr<VulkanContext> pCtx;
    SharedPtr<VulkanCommandPool> pCmdPool;
    vk::CommandBufferLevel mLevel;

    ::std::vector<vk::CommandBuffer> mCmdBuffer;
};