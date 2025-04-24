#include "RenderFrame.h"

#include "Core/Vulkan/Native/Buffer.h"
#include "RenderResourceManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderFrame::RenderFrame(VulkanContext& context,
                         RenderResourceManager& renderResMgr, uint32_t idx)
    : mContext(context),
      mRenderResMgr(renderResMgr),
      mIdx(idx),
      mFencePool(MakeUnique<FencePool>(context)),
      mSemaphorePool(MakeUnique<SemaphorePool>(context)),
      mPresentFinished(MakeUnique<Semaphore>(context)),
      mRenderFinished(MakeUnique<Semaphore>(context)),
      mSwapchainPresent(MakeUnique<Semaphore>(context)),
      mQueryPool(MakeUnique<QueryPool>(context)) {
    auto indexMap = mContext.GetDevice().GetQueueFamilyIndices();

    for (auto const& [type, index] : indexMap) {
        mCmdPools.emplace(index, MakeUnique<CommandPool>(mContext, index));
    }

    mOutOfBoundsCheckBufferName = "Frame Readback Buffer ";
    mOutOfBoundsCheckBufferName =
        mOutOfBoundsCheckBufferName + ::std::to_string(mIdx).c_str();
    mRenderResMgr.CreateBuffer(
        mOutOfBoundsCheckBufferName.c_str(), 4 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        Buffer::MemoryType::ReadBack);

    mModelIDBufferName = "ModelID Readback Buffer";
    mModelIDBufferName = mModelIDBufferName + ::std::to_string(mIdx).c_str();
    mRenderResMgr.CreateBuffer_ScreenSizeRelated(
        mModelIDBufferName.c_str(), 1600 * 900 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst,
        Buffer::MemoryType::ReadBack, sizeof(uint32_t));

    mGlobalUniformBufferName = "GlobalUniformBuffer";
    mGlobalUniformBufferName =
        mGlobalUniformBufferName + ::std::to_string(mIdx).c_str();
}

void RenderFrame::PrepareBindlessDescPool(
    Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
    vk::DescriptorType type) {
    mBindlessDescPool = MakeShared<BindlessDescPool>(mContext, pso, type);
}

void RenderFrame::PrepareGlobalUniformBuffer(uint32_t size) {
    mRenderResMgr.CreateBuffer(
        mGlobalUniformBufferName.c_str(), size,
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferSrc,
        Buffer::MemoryType::Staging);
}

uint32_t RenderFrame::GetIndex() const {
    return mIdx;
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

Semaphore const& RenderFrame::GetPresentFinishedSemaphore() const {
    return *mPresentFinished;
}

Semaphore const& RenderFrame::GetRenderFinishedSemaphore() const {
    return *mRenderFinished;
}

Semaphore const& RenderFrame::GetSwapchainPresentSemaphore() const {
    return *mSwapchainPresent;
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

QueryPool& RenderFrame::GetQueryPool() const {
    return *mQueryPool;
}

void RenderFrame::Reset() {
    VK_CHECK(mFencePool->Wait());
    VK_CHECK(mFencePool->Reset());

    for (auto const& [_, cp] : mCmdPools) {
        cp->Reset();
    }

    mSemaphorePool->Reset();
}

void RenderFrame::CullRegister(
    SharedPtr<IntelliDesign_NS::Core::SceneGraph::Node> const& node) {
    mNodes.push_back(node);
}

void RenderFrame::ClearNodes() {
    mNodes.clear();
}

Type_STLVector<SharedPtr<IntelliDesign_NS::Core::SceneGraph::Node>>&
RenderFrame::GetInFrustumNodes() {
    return mNodes;
}

const char* RenderFrame::GetOutOfBoundsCheckBufferName() const {
    return mOutOfBoundsCheckBufferName.c_str();
}

RenderResource const& RenderFrame::GetOutOfBoundsCheckBuffer() const {
    return mRenderResMgr[mOutOfBoundsCheckBufferName.c_str()];
}

const char* RenderFrame::GetModelIDBufferName() const {
    return mModelIDBufferName.c_str();
}

const char* RenderFrame::GetGlobalUniformBufferName() const {
    return mGlobalUniformBufferName.c_str();
}

}  // namespace IntelliDesign_NS::Vulkan::Core