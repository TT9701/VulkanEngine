#include "RenderFrame.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderFrame::RenderFrame(VulkanContext& context)
    : mContext(context),
      mFencePool(MakeUnique<FencePool>(context)),
      mSemaphorePool(MakeUnique<SemaphorePool>(context)),
      mReady4Render(MakeUnique<Semaphore>(context)),
      mReady4Present(MakeUnique<Semaphore>(context)) {
    auto indexMap = mContext.GetDevice().GetQueueFamilyIndices();

    for (auto const& [type, index] : indexMap) {
        mCmdPools.emplace(index, MakeUnique<CommandPool>(mContext, index));
    }
}

void RenderFrame::PrepareBindlessDescPool(
    Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
    vk::DescriptorType type) {
    mBindlessDescPool = MakeShared<BindlessDescPool>(mContext, pso, type);
}

FencePool& RenderFrame::GetFencePool() const {
    return *mFencePool;
}

SemaphorePool& RenderFrame::GetSemaphorePool() const {
    return *mSemaphorePool;
}

vk::Semaphore RenderFrame::RequestSemaphore() {
    return mSemaphorePool->RequestSemaphore();
}

vk::Semaphore RenderFrame::RequestSemaphore_WithOwnership() {
    return mSemaphorePool->RequestSemaphore_WithOwnership();
}

void RenderFrame::ReleaseOwnedSemaphore(vk::Semaphore semaphore) {
    mSemaphorePool->ReleaseOwnedSemaphore(semaphore);
}

vk::Fence RenderFrame::RequestFence(vk::FenceCreateFlags flags) {
    return mFencePool->RequestFence(flags);
}

Semaphore const& RenderFrame::GetReady4RenderSemaphore() const {
    return *mReady4Render;
}

Semaphore const& RenderFrame::GetReady4PresentSemaphore() const {
    return *mReady4Present;
}

CmdBufferToBegin RenderFrame::GetGraphicsCmdBuf() const {
    return {mCmdPools
                .at(mContext.GetDevice().GetQueueFamilyIndex(
                    vk::QueueFlagBits::eGraphics))
                ->RequestCommandBuffer()};
}

CmdBufferToBegin RenderFrame::GetComputeCmdBuf() const {
    return {mCmdPools
                .at(mContext.GetDevice().GetQueueFamilyIndex(
                    vk::QueueFlagBits::eCompute))
                ->RequestCommandBuffer()};
}

CmdBufferToBegin RenderFrame::GetTransferCmdBuf() const {
    return {mCmdPools
                .at(mContext.GetDevice().GetQueueFamilyIndex(
                    vk::QueueFlagBits::eTransfer))
                ->RequestCommandBuffer()};
}

BindlessDescPool& RenderFrame::GetBindlessDescPool() const {
    return *mBindlessDescPool;
}

void RenderFrame::Reset() {
    VK_CHECK(mFencePool->Wait());
    VK_CHECK(mFencePool->Reset());

    for (auto const& [_, cp] : mCmdPools) {
        cp->Reset();
    }

    mSemaphorePool->Reset();
}

}  // namespace IntelliDesign_NS::Vulkan::Core