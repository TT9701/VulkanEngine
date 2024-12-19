#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

namespace IntelliDesign_NS::Vulkan::Core {

enum class RenderGraphQueueType { Graphics, Compute, AsyncCompute };

class VulkanContext;
class RenderResourceManager;
class PipelineManager;
class Swapchain;
class RenderPassBindingInfo_PSO;

class RenderGraph {

public:
    RenderGraph(VulkanContext& context, RenderResourceManager& resMgr,
                PipelineManager& pipelineMgr, DescriptorSetPool& descPool,
                Swapchain& sc);

    ~RenderGraph() = default;

    CLASS_NO_COPY_MOVE(RenderGraph);

    RenderPassBindingInfo_PSO& AddRenderPass(const char* name,
                                             RenderGraphQueueType type);

    RenderPassBindingInfo_PSO& FindRenderPass(const char* name);

    friend RenderPassBindingInfo_PSO;

private:
    VulkanContext& mContext;
    RenderResourceManager& mResMgr;
    PipelineManager& mPipelineMgr;
    DescriptorSetPool& mDescPool;
    Swapchain& mSwapchain;

    Type_STLUnorderedMap_String<uint32_t> mPassNameToIndex;
    Type_STLVector<UniquePtr<RenderPassBindingInfo_PSO>> mPasses;

    Type_STLUnorderedMap_String<uint32_t> mResNameToIndex;
    // TODO: 将 RenderResourcce 的管理放在这里，而不是引用
    Type_STLVector<RenderResource*> mRenderResources;
};

}  // namespace IntelliDesign_NS::Vulkan::Core