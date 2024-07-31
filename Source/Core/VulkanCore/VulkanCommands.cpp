#include "VulkanCommands.hpp"

#include <utility>

#include "VulkanContext.hpp"

VulkanCommandPool::VulkanCommandPool(VulkanContext* ctx,
                                     uint32_t       queueFamilysIndex,
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

VulkanCommandBuffers::VulkanCommandBuffers(VulkanContext*         ctx,
                                           VulkanCommandPool*     pool,
                                           uint32_t               count,
                                           vk::CommandBufferLevel level)
    : pContex(ctx),
      pCmdPool(pool),
      mLevel(level),
      mCmdBuffer(CreateCommandBuffers(count)) {}

::std::vector<vk::CommandBuffer> VulkanCommandBuffers::CreateCommandBuffers(
    uint32_t count) {
    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(pCmdPool->GetHandle())
        .setLevel(mLevel)
        .setCommandBufferCount(count);

    return pContex->GetDeviceHandle().allocateCommandBuffers(cmdAllocInfo);
}