#include "VulkanCommands.hpp"

#include <utility>

#include "VulkanContext.hpp"

VulkanCommandPool::VulkanCommandPool(SharedPtr<VulkanContext> ctx,
                                     uint32_t queueFamilysIndex,
                                     vk::CommandPoolCreateFlags flags)
    : pCtx(std::move(ctx)),
      mFlags(flags),
      mQueueFamilysIndex(queueFamilysIndex),
      mCmdPool(CreateCommandPool()) {}

VulkanCommandPool::~VulkanCommandPool() {
    pCtx->GetDevice()->GetHandle().destroy(mCmdPool);
}

vk::CommandPool VulkanCommandPool::CreateCommandPool() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.setFlags(mFlags).setQueueFamilyIndex(mQueueFamilysIndex);

    return pCtx->GetDevice()->GetHandle().createCommandPool(cmdPoolCreateInfo);
}

VulkanCommandBuffer::VulkanCommandBuffer(
    SharedPtr<VulkanContext> ctx, SharedPtr<VulkanCommandPool> pool,
    vk::CommandBufferLevel level, uint32_t count)
    : pCtx(std::move(ctx)),
      pCmdPool(std::move(pool)),
      mLevel(level),
      mCmdBuffer(CreateCommandBuffer(count)) {}

::std::vector<vk::CommandBuffer> VulkanCommandBuffer::CreateCommandBuffer(
    uint32_t count) {
    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(pCmdPool->GetHandle())
        .setLevel(mLevel)
        .setCommandBufferCount(count);

    return pCtx->GetDevice()->GetHandle().allocateCommandBuffers(cmdAllocInfo);
}