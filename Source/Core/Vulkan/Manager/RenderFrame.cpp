#include "RenderFrame.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderFrame::RenderFrame(Context* context)
    : pContext(context),
      mFencePool(MakeUnique<FencePool>(context)),
      mReady4Render(MakeUnique<Semaphore>(context)),
      mReady4Present(MakeUnique<Semaphore>(context)) {
    ::std::vector<::std::optional<uint32_t>> indices {};
    indices.emplace_back(
        pContext->GetPhysicalDevice().GetGraphicsQueueFamilyIndex());
    indices.emplace_back(
        pContext->GetPhysicalDevice().GetComputeQueueFamilyIndex());
    indices.emplace_back(
        pContext->GetPhysicalDevice().GetTransferQueueFamilyIndex());

    for (auto const& idx : indices) {
        if (idx) {
            auto value = *idx;
            mCmdPools.emplace(value, MakeUnique<CommandPool>(pContext, value));
        }
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
    auto gfxIdx = *pContext->GetPhysicalDevice().GetGraphicsQueueFamilyIndex();
    return {mCmdPools.at(gfxIdx)->RequestCommandBuffer()};
}

CmdBufferToBegin RenderFrame::GetCmpCmdBuf() const {
    if (auto cmpIdx =
            pContext->GetPhysicalDevice().GetComputeQueueFamilyIndex()) {
        return {mCmdPools.at(cmpIdx.value())->RequestCommandBuffer()};
    } else {
        return {mCmdPools
                    .at(*pContext->GetPhysicalDevice()
                             .GetGraphicsQueueFamilyIndex())
                    ->RequestCommandBuffer()};
    }
}

CmdBufferToBegin RenderFrame::GetTsfCmdBuf() const {
    if (auto tsfIdx =
            pContext->GetPhysicalDevice().GetTransferQueueFamilyIndex()) {
        return {mCmdPools.at(tsfIdx.value())->RequestCommandBuffer()};
    } else {
        return {mCmdPools
                    .at(*pContext->GetPhysicalDevice()
                             .GetGraphicsQueueFamilyIndex())
                    ->RequestCommandBuffer()};
    }
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