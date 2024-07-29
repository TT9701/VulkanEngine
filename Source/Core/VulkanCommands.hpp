#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanCommandPool {
public:
    VulkanCommandPool(VulkanContext* ctx, uint32_t queueFamilysIndex,
                      vk::CommandPoolCreateFlags flags =
                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    ~VulkanCommandPool();
    MOVABLE_ONLY(VulkanCommandPool);

public:
    vk::CommandPool GetHandle() const { return mCmdPool; }

private:
    vk::CommandPool CreateCommandPool();

private:
    VulkanContext*             pCtx;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t                   mQueueFamilysIndex;

    vk::CommandPool mCmdPool;
};

class VulkanCommandBuffers {
public:
    VulkanCommandBuffers(
        VulkanContext* ctx, VulkanCommandPool* pool, uint32_t count = 1u,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    ~VulkanCommandBuffers() = default;
    MOVABLE_ONLY(VulkanCommandBuffers);

public:
    vk::CommandBuffer GetHandle(uint32_t index = 0) const {
        return mCmdBuffer[index];
    }

    uint32_t GetBufferCount() const {
        return static_cast<uint32_t>(mCmdBuffer.size());
    }

private:
    ::std::vector<vk::CommandBuffer> CreateCommandBuffers(uint32_t count);

private:
    VulkanContext*         pContex;
    VulkanCommandPool*     pCmdPool;
    vk::CommandBufferLevel mLevel;

    ::std::vector<vk::CommandBuffer> mCmdBuffer;
};