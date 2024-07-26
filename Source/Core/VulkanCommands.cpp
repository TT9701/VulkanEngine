#include "VulkanCommands.hpp"

#include <utility>

#include "VulkanContext.hpp"

VulkanCommandPool::VulkanCommandPool(Type_SPInstance<VulkanContext> ctx,
                                     uint32_t queueFamilysIndex,
                                     vk::CommandPoolCreateFlags flags)
    : pCtx(std::move(ctx)),
      mFlags(flags),
      mQueueFamilysIndex(queueFamilysIndex),
      mCmdPool(CreateCommandPool()) {}

VulkanCommandPool::~VulkanCommandPool() {
    pCtx->GetDeviceHandle().destroy(mCmdPool);
}

vk::CommandPool VulkanCommandPool::CreateCommandPool() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.setFlags(mFlags).setQueueFamilyIndex(mQueueFamilysIndex);

    return pCtx->GetDeviceHandle().createCommandPool(cmdPoolCreateInfo);
}

VulkanCommandBuffer::VulkanCommandBuffer(
    Type_SPInstance<VulkanContext> ctx, Type_SPInstance<VulkanCommandPool> pool,
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

    return pCtx->GetDeviceHandle().allocateCommandBuffers(cmdAllocInfo);
}