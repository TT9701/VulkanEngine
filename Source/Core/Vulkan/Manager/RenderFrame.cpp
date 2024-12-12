#include "RenderFrame.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderFrame::RenderFrame(VulkanContext* context)
    : pContext(context),
      mFencePool(MakeUnique<FencePool>(*context)),
      mReady4Render(MakeUnique<Semaphore>(context)),
      mReady4Present(MakeUnique<Semaphore>(context)) {
    auto indexMap = pContext->GetDevice().GetQueueFamilyIndices();

    for (auto const& [type, index] : indexMap) {
        mCmdPools.emplace(index, MakeUnique<CommandPool>(*pContext, index));
    }
}

void RenderFrame::PrepareBindlessDescPool(
    Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
    vk::DescriptorType type) {
    mBindlessDescPool = MakeShared<BindlessDescPool>(pContext, pso, type);
}

FencePool& RenderFrame::GetFencePool() const {
    return *mFencePool;
}

Semaphore const& RenderFrame::GetReady4RenderSemaphore() const {
    return *mReady4Render;
}

Semaphore const& RenderFrame::GetReady4PresentSemaphore() const {
    return *mReady4Present;
}

CmdBufferToBegin RenderFrame::GetGfxCmdBuf() const {
    return {mCmdPools
                .at(pContext->GetDevice().GetQueueFamilyIndex(
                    vk::QueueFlagBits::eGraphics))
                ->RequestCommandBuffer()};
}

CmdBufferToBegin RenderFrame::GetCmpCmdBuf() const {
    return {mCmdPools
                .at(pContext->GetDevice().GetQueueFamilyIndex(
                    vk::QueueFlagBits::eCompute))
                ->RequestCommandBuffer()};
}

CmdBufferToBegin RenderFrame::GetTsfCmdBuf() const {
    return {mCmdPools
                .at(pContext->GetDevice().GetQueueFamilyIndex(
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
}

}  // namespace IntelliDesign_NS::Vulkan::Core