#include "RenderSequence.h"

#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Native/Swapchain.h"
#include "RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderSequence::RenderSequence(VulkanContext& context,
                               RenderResourceManager& resMgr,
                               PipelineManager& pipelineMgr,
                               DescriptorSetPool& descPool, Swapchain& sc)
    : mContext(context),
      mResMgr(resMgr),
      mPipelineMgr(pipelineMgr),
      mDescPool(descPool),
      mSwapchain(sc) {}

RenderPassBindingInfo_PSO& RenderSequence::AddRenderPass(
    const char* name, RenderGraphQueueType type) {
    if (mPassNameToIndex.contains(name)) {
        return *mPasses[mPassNameToIndex.at(name)].pso;
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(
            MakeUnique<RenderPassBindingInfo_PSO>(*this, index, type));
        mPasses.back().pso->SetName(name);
        mPassNameToIndex[name] = index;

        mPassBarrierInfos.emplace_back();

        return *mPasses.back().pso;
    }
}

RenderPassBindingInfo_PSO& RenderSequence::FindRenderPass(const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return *mPasses[mPassNameToIndex.at(name)].pso;
    } else {
        throw ::std::runtime_error("invalid render pass name!");
    }
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
        if (n == "_Swapchain_") {
            mResNameToIndex["_Swapchain_"] = mRenderResources.size();
            mRenderResources.push_back(&mSwapchain.GetCurrentImage());
        } else {
            mResNameToIndex[name] = mRenderResources.size();
            mRenderResources.push_back(&mResMgr[name]);
        }
        return mRenderResources.size() - 1;
    }
}

void RenderSequence::GenerateBarriers() {
    uint32_t passCount = mPasses.size();
    for (uint32_t i = 0; i < passCount; ++i) {
        for (auto const& flush : mPassBarrierInfos[i].flush) {
            uint32_t flushResIdx = flush.resourceIndex;
            bool foundInvalidate {false};
            Barrier invalidateBarrier {};
            for (uint32_t it = (i + passCount - 1) % passCount; it != i;
                 it = (it + passCount - 1) % passCount) {
                if (foundInvalidate)
                    break;
                for (auto const& invalidate :
                     mPassBarrierInfos[it].invalidate) {
                    uint32_t invalidateResIdx = invalidate.resourceIndex;
                    if (flushResIdx == invalidateResIdx) {
                        foundInvalidate = true;
                        invalidateBarrier = invalidate;
                        break;
                    }
                }
            }
            if (foundInvalidate) {
                if (mPasses[i].barrier == nullptr) {
                    mPasses[i].barrier =
                        MakeUnique<RenderPassBindingInfo_Barrier>(mContext,
                                                                  mResMgr);
                }
                auto& barrier = *mPasses[i].barrier;

                auto resource = mRenderResources[flushResIdx];
                auto resType = resource->GetType();

                if (resType == RenderResource::Type::Buffer) {
                    barrier.AddBufferBarrier(
                        Type_STLString {resource->GetName()}.c_str(),
                        {.srcStageMask = invalidateBarrier.stages,
                         .srcAccessMask = invalidateBarrier.access,
                         .dstStageMask = flush.stages,
                         .dstAccessMask = flush.access});
                } else {

                    barrier.AddImageBarrier(
                        Type_STLString {resource->GetName()}.c_str(),
                        {.srcStageMask = invalidateBarrier.stages,
                         .srcAccessMask = invalidateBarrier.access,
                         .dstStageMask = flush.stages,
                         .dstAccessMask = flush.access,
                         .oldLayout = invalidateBarrier.layout,
                         .newLayout = flush.layout,
                         .aspect = resource->GetTexUsage()
                                         & vk::ImageUsageFlagBits::
                                               eDepthStencilAttachment
                                     ? vk::ImageAspectFlagBits::eDepth
                                     : vk::ImageAspectFlagBits::eColor});
                }
            }
        }
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core