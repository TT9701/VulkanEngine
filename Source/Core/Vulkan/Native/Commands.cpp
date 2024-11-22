#include "Commands.h"

#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

CommandPool::CommandPool(Context* ctx, uint32_t queueFamilysIndex,
                         vk::CommandPoolCreateFlags flags)
    : pCtx(ctx),
      mFlags(flags),
      mQueueFamilysIndex(queueFamilysIndex),
      mCmdPool(CreateCommandPool()) {}

CommandPool::~CommandPool() {
    pCtx->GetDeviceHandle().destroy(mCmdPool);
}

CommandBuffer& CommandPool::RequestCommandBuffer() {
    if (mActiveCmdBufCount < mCmdBuffers.size()) {
        return *mCmdBuffers[mActiveCmdBufCount++];
    }

    mCmdBuffers.emplace_back(MakeUnique<CommandBuffer>(pCtx, this));

    mActiveCmdBufCount++;

    return *mCmdBuffers.back();
}

void CommandPool::Reset() {
    mActiveCmdBufCount = 0;
}

vk::CommandPool CommandPool::CreateCommandPool() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.setFlags(mFlags).setQueueFamilyIndex(mQueueFamilysIndex);

    return pCtx->GetDeviceHandle().createCommandPool(cmdPoolCreateInfo);
}

CommandBuffer::CommandBuffer(Context* ctx, CommandPool* pool,
                             vk::CommandBufferLevel level)
    : pContex(ctx),
      pCmdPool(pool),
      mLevel(level),
      mCmdBuffer(CreateCommandBuffer()) {}

void CommandBuffer::Reset() {
    mCmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
}

void CommandBuffer::End() {
    mCmdBuffer.end();
}

vk::CommandBuffer CommandBuffer::CreateCommandBuffer() {
    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(pCmdPool->GetHandle())
        .setLevel(mLevel)
        .setCommandBufferCount(1);

    auto vec = pContex->GetDeviceHandle().allocateCommandBuffers(cmdAllocInfo);
    return vec.front();
}

CmdBufferToBegin::CmdBufferToBegin(CommandBuffer& cmd) : mBuffer(cmd) {
    // mBuffer.Reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    mBuffer.GetHandle().begin(beginInfo);
}

vk::CommandBuffer CmdBufferToBegin::GetHandle() const {
    return mBuffer.GetHandle();
}

void CmdBufferToBegin::End() {
    mBuffer.End();
}

}  // namespace IntelliDesign_NS::Vulkan::Core