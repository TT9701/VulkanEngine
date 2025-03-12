#include "RenderSequence.h"

#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderSequence::RenderSequence(VulkanContext& context,
                               RenderResourceManager& resMgr,
                               PipelineManager& pipelineMgr,
                               DescriptorSetPool& descPool)
    : mContext(context),
      mResMgr(resMgr),
      mPipelineMgr(pipelineMgr),
      mDescPool(descPool) {}

RenderPassBindingInfo_PSO& RenderSequence::AddRenderPass(const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return *dynamic_cast<RenderPassBindingInfo_PSO*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(
            *this, MakeUnique<RenderPassBindingInfo_PSO>(*this, index));
        mPassNameToIndex[name] = index;

        mPassBarrierInfos.emplace_back();

        auto ptr = dynamic_cast<RenderPassBindingInfo_PSO*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
        ptr->SetName(name);

        return *ptr;
    }
}

RenderPassBindingInfo_PSO& RenderSequence::AddRenderPass(
    const char* name, RenderResource const* dgcSeqBuf) {
    if (mPassNameToIndex.contains(name)) {
        return *dynamic_cast<RenderPassBindingInfo_PSO*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(*this, MakeUnique<RenderPassBindingInfo_PSO>(
                                        *this, index, dgcSeqBuf));
        mPassNameToIndex[name] = index;

        mPassBarrierInfos.emplace_back();

        auto ptr = dynamic_cast<RenderPassBindingInfo_PSO*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
        ptr->SetName(name);

        return *ptr;
    }
}

RenderPassBindingInfo_Copy& RenderSequence::AddCopyPass(const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return *dynamic_cast<RenderPassBindingInfo_Copy*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(
            *this, MakeUnique<RenderPassBindingInfo_Copy>(*this, index));
        mPassNameToIndex[name] = index;
        mPassBarrierInfos.emplace_back();

        auto ptr = dynamic_cast<RenderPassBindingInfo_Copy*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());

        return *ptr;
    }
}

RenderPassBindingInfo_Executor& RenderSequence::AddExecutor(const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return *dynamic_cast<RenderPassBindingInfo_Executor*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(
            *this, MakeUnique<RenderPassBindingInfo_Executor>(*this, index));
        mPassNameToIndex[name] = index;
        mPassBarrierInfos.emplace_back();
        auto ptr = dynamic_cast<RenderPassBindingInfo_Executor*>(
            mPasses[mPassNameToIndex.at(name)].binding.get());
        return *ptr;
    }
}

RenderSequence::RenderPassBindingInfo& RenderSequence::FindPass(
    const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return mPasses[mPassNameToIndex.at(name)];
    } else {
        throw ::std::runtime_error("invalid render pass name!");
    }
}

RenderSequence::RenderPassBindingInfo&
RenderSequence::GetRenderToSwapchainPass() {
    return mPasses.back();
}

void RenderSequence::RecordPass(const char* name, vk::CommandBuffer cmd) {
    auto& pass = mPasses[mPassNameToIndex[name]];
    pass.RecordCmd(cmd);
}

uint32_t RenderSequence::AddRenderResource(const char* name) {
    Type_STLString n {name};
    auto it = std::ranges::find_if(
        mResNameToIndex, [&n](::std::pair<Type_STLString, uint32_t> const& p) {
            return p.first == n;
        });
    if (it != mResNameToIndex.end()) {
        return it->second;
    } else {
        mResNameToIndex[name] = mRenderResources.size();
        mRenderResources.emplace_back(&mResMgr[name]);
        return mRenderResources.size() - 1;
    }
}

void RenderSequence::RenderPassBindingInfo::RecordCmd(vk::CommandBuffer cmd) {
    if (preBarrieres)
        preBarrieres->RecordCmd(cmd);

    binding->RecordCmd(cmd);

    if (postBarrieres)
        postBarrieres->RecordCmd(cmd);
}

void RenderSequence::RenderPassBindingInfo::Update(
    Type_STLVector<Type_STLString> const& resNames) {
    if (preBarrieres)
        preBarrieres->Update(resNames);

    binding->Update(resNames);

    if (postBarrieres)
        postBarrieres->Update(resNames);
}

void RenderSequence::RenderPassBindingInfo::OnResize(vk::Extent2D extent) {
    if (preBarrieres)
        preBarrieres->OnResize(extent);

    binding->OnResize(extent);

    if (postBarrieres)
        postBarrieres->OnResize(extent);
}

void RenderSequence::GenerateBarriers() {
    uint32_t passCount = mPasses.size();
    for (uint32_t i = 0; i < passCount; ++i) {
        for (auto const& invalidate : mPassBarrierInfos[i]) {
            uint32_t invalidateResIdx = invalidate.resourceIndex;

            bool foundFlush {false};
            Barrier flushBarrier {};
            for (uint32_t it = (i + passCount - 1) % passCount; it != i;
                 it = (it + passCount - 1) % passCount) {
                if (foundFlush)
                    break;
                for (auto const& flush : mPassBarrierInfos[it]) {
                    uint32_t flushResIdx = flush.resourceIndex;
                    if (flushResIdx == invalidateResIdx) {
                        foundFlush = true;
                        flushBarrier = flush;
                        break;
                    }
                }
            }
            if (foundFlush) {
                if (mPasses[i].preBarrieres == nullptr) {
                    mPasses[i].preBarrieres =
                        MakeUnique<RenderPassBindingInfo_Barrier>(mContext,
                                                                  mResMgr);
                }
                auto& barrier = *mPasses[i].preBarrieres;

                auto resource = mRenderResources[invalidateResIdx];
                auto resType = resource->GetType();

                if (resType == RenderResource::Type::Buffer) {
                    barrier.AddBufferBarrier(
                        Type_STLString {resource->GetName()}.c_str(),
                        {.srcStageMask = flushBarrier.stages,
                         .srcAccessMask = flushBarrier.access,
                         .dstStageMask = invalidate.stages,
                         .dstAccessMask = invalidate.access});
                } else {

                    barrier.AddImageBarrier(
                        Type_STLString {resource->GetName()}.c_str(),
                        {.srcStageMask = flushBarrier.stages,
                         .srcAccessMask = flushBarrier.access,
                         .dstStageMask = invalidate.stages,
                         .dstAccessMask = invalidate.access,
                         .oldLayout = flushBarrier.layout,
                         .newLayout = invalidate.layout,
                         .aspect = resource->GetTexUsage()
                                         & vk::ImageUsageFlagBits::
                                               eDepthStencilAttachment
                                     ? vk::ImageAspectFlagBits::eDepth
                                           | vk::ImageAspectFlagBits::eStencil
                                     : vk::ImageAspectFlagBits::eColor});
                }
            }
        }
    }

    // Generate
    for (uint32_t i = 0; i < passCount; ++i) {
        if (mPasses[i].preBarrieres)
            mPasses[i].preBarrieres->GenerateMetaData();
        if (mPasses[i].postBarrieres)
            mPasses[i].postBarrieres->GenerateMetaData();
    }

    GeneratePreRenderBarriers();
}

void RenderSequence::GeneratePreRenderBarriers() {
    mPreRenderBarrieres.reset();

    ::std::unordered_map<uint32_t, Barrier> resIdxToBarInfoMap {};
    uint32_t passCount = mPasses.size();
    for (uint32_t i = 0; i < passCount; ++i) {
        for (auto const& invalidate : mPassBarrierInfos[i]) {
            uint32_t invalidateResIdx = invalidate.resourceIndex;
            if (resIdxToBarInfoMap.contains(invalidateResIdx))
                continue;

            resIdxToBarInfoMap[invalidateResIdx] = invalidate;
        }
    }

    if (!resIdxToBarInfoMap.empty()) {
        mPreRenderBarrieres =
            MakeUnique<RenderPassBindingInfo_Barrier>(mContext, mResMgr);
    }
    for (auto const& [resIdx, barInfo] : resIdxToBarInfoMap) {
        auto resource = mRenderResources[resIdx];
        auto resType = resource->GetType();

        if (resType == RenderResource::Type::Buffer) {
            mPreRenderBarrieres->AddBufferBarrier(
                Type_STLString {resource->GetName()}.c_str(),
                {.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                 .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                 .dstStageMask = barInfo.stages,
                 .dstAccessMask = barInfo.access});
        } else {

            mPreRenderBarrieres->AddImageBarrier(
                Type_STLString {resource->GetName()}.c_str(),
                {.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                 .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                 .dstStageMask = barInfo.stages,
                 .dstAccessMask = barInfo.access,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = barInfo.layout == vk::ImageLayout::eGeneral
                                ? vk::ImageLayout::eShaderReadOnlyOptimal
                                : barInfo.layout,
                 .aspect =
                     resource->GetTexUsage()
                             & vk::ImageUsageFlagBits::eDepthStencilAttachment
                         ? vk::ImageAspectFlagBits::eDepth
                               | vk::ImageAspectFlagBits::eStencil
                         : vk::ImageAspectFlagBits::eColor});
        }
    }

    mPreRenderBarrieres->GenerateMetaData();
}

void RenderSequence::ExecutePreRenderBarriers() {
    if (mPreRenderBarrieres) {
        auto cmd = mContext.CreateCmdBufToBegin(
            mContext.GetQueue(QueueType::Graphics));
        mPreRenderBarrieres->RecordCmd(cmd.mHandle);
    }
}

void RenderSequence::Clear() {
    mPassNameToIndex.clear();
    mPasses.clear();
    mResNameToIndex.clear();
    mRenderResources.clear();
    mPassBarrierInfos.clear();
}

}  // namespace IntelliDesign_NS::Vulkan::Core