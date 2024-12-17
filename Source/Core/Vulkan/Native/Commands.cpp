#include "Commands.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

CommandPool::CommandPool(VulkanContext& ctx, uint32_t queueFamilysIndex,
                         vk::CommandPoolCreateFlags flags)
    : mContext(ctx),
      mFlags(flags),
      mQueueFamilysIndex(queueFamilysIndex),
      mHandle(CreateCommandPool()) {}

CommandPool::~CommandPool() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::CommandPool CommandPool::GetHandle() const {
    return mHandle;
}

CommandBuffer& CommandPool::RequestCommandBuffer(vk::CommandBufferLevel level) {
    if (mActiveCmdBufCount < mCmdBuffers.size()) {
        return *mCmdBuffers[mActiveCmdBufCount++];
    }

    mCmdBuffers.emplace_back(MakeUnique<CommandBuffer>(mContext, *this, level));

    mActiveCmdBufCount++;

    return *mCmdBuffers.back();
}

void CommandPool::Reset() {
    mActiveCmdBufCount = 0;
}

vk::CommandPool CommandPool::CreateCommandPool() {
    vk::CommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.setFlags(mFlags).setQueueFamilyIndex(mQueueFamilysIndex);

    return mContext.GetDevice()->createCommandPool(cmdPoolCreateInfo);
}

CommandBuffer::CommandBuffer(VulkanContext& ctx, CommandPool& pool,
                             vk::CommandBufferLevel level)
    : mContex(ctx),
      mCmdPool(pool),
      mLevel(level),
      mHandle(CreateCommandBuffer()) {}

vk::CommandBuffer CommandBuffer::GetHandle() const {
    return mHandle;
}

vk::CommandBuffer const* CommandBuffer::operator->() const {
    return &mHandle;
}

void CommandBuffer::Reset() {
    mHandle.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
}

void CommandBuffer::End() {
    mHandle.end();
}

vk::CommandBuffer CommandBuffer::CreateCommandBuffer() {
    vk::CommandBufferAllocateInfo cmdAllocInfo {};
    cmdAllocInfo.setCommandPool(mCmdPool.GetHandle())
        .setLevel(mLevel)
        .setCommandBufferCount(1);

    auto vec = mContex.GetDevice()->allocateCommandBuffers(cmdAllocInfo);
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