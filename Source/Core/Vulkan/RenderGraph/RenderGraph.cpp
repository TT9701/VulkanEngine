#include "RenderGraph.h"

#include "RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderGraph::RenderGraph(VulkanContext& context, RenderResourceManager& resMgr,
                         PipelineManager& pipelineMgr,
                         DescriptorSetPool& descPool, Swapchain& sc)
    : mContext(context),
      mResMgr(resMgr),
      mPipelineMgr(pipelineMgr),
      mDescPool(descPool),
      mSwapchain(sc) {}

RenderPassBindingInfo_PSO& RenderGraph::AddRenderPass(
    const char* name, RenderGraphQueueType type) {
    if (mPassNameToIndex.contains(name)) {
        return *mPasses[mPassNameToIndex.at(name)];
    } else {
        uint32_t index = mPasses.size();
        mPasses.emplace_back(
            MakeUnique<RenderPassBindingInfo_PSO>(*this, index, type));
        mPasses.back()->SetName(name);
        mPassNameToIndex[name] = index;
        return *mPasses.back();
    }
}

RenderPassBindingInfo_PSO& RenderGraph::FindRenderPass(const char* name) {
    if (mPassNameToIndex.contains(name)) {
        return *mPasses[mPassNameToIndex.at(name)];
    } else {
        throw ::std::runtime_error("invalid render pass name!");
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core