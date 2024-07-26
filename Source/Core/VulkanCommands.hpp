#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanCommandPool {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanCommandPool(Type_SPInstance<VulkanContext> ctx,
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
    Type_SPInstance<VulkanContext> pCtx;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t mQueueFamilysIndex;

    vk::CommandPool mCmdPool;
};

class VulkanCommandBuffer {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanCommandBuffer(
        Type_SPInstance<VulkanContext> ctx,
        Type_SPInstance<VulkanCommandPool> pool,
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
    Type_SPInstance<VulkanContext> pCtx;
    Type_SPInstance<VulkanCommandPool> pCmdPool;
    vk::CommandBufferLevel mLevel;

    ::std::vector<vk::CommandBuffer> mCmdBuffer;
};