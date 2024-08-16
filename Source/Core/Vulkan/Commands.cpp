#include "Commands.hpp"

#include <utility>

#include "Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

CommandPool::CommandPool(Context* ctx,
                                     uint32_t queueFamilysIndex,
                                     vk::CommandPoolCreateFlags flags)
    : pCtx(std::move(ctx)),
      mFlags(flags),
      mQueueFamilysIndex(queueFamilysIndex),
      mCmdPool(CreateCommandPool()) {}

CommandPool::~CommandPool() {
    pCtx->GetDeviceHandle().destroy(mCmdPool);
}

vk::CommandPool CommandPool::CreateCommandPool() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.setFlags(mFlags).setQueueFamilyIndex(mQueueFamilysIndex);

    return pCtx->GetDeviceHandle().createCommandPool(cmdPoolCreateInfo);
}

CommandBuffers::CommandBuffers(Context* ctx,
                                           CommandPool* pool,
                                           uint32_t count,
                                           vk::CommandBufferLevel level)
    : pContex(ctx),
      pCmdPool(pool),
      mLevel(level),
      mCmdBuffer(CreateCommandBuffers(count)) {}

::std::vector<vk::CommandBuffer> CommandBuffers::CreateCommandBuffers(
    uint32_t count) {
    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(pCmdPool->GetHandle())
        .setLevel(mLevel)
        .setCommandBufferCount(count);

    return pContex->GetDeviceHandle().allocateCommandBuffers(cmdAllocInfo);
}

}  // namespace IntelliDesign_NS::Vulkan::Core