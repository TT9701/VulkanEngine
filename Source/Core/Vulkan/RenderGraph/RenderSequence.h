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
class RenderPassBindingInfo_Barrier;

class RenderSequenceConfig {
public:


};

class RenderSequence {

public:
    RenderSequence(VulkanContext& context, RenderResourceManager& resMgr,
                   PipelineManager& pipelineMgr, DescriptorSetPool& descPool,
                   Swapchain& sc);

    ~RenderSequence() = default;

    CLASS_NO_COPY_MOVE(RenderSequence);

    RenderPassBindingInfo_PSO& AddRenderPass(const char* name,
                                             RenderGraphQueueType type);

    RenderPassBindingInfo_PSO& FindRenderPass(const char* name);

    void GenerateBarriers();

    friend RenderPassBindingInfo_PSO;

private:
    uint32_t AddRenderResource(const char* name);

    VulkanContext& mContext;
    RenderResourceManager& mResMgr;
    PipelineManager& mPipelineMgr;
    DescriptorSetPool& mDescPool;
    Swapchain& mSwapchain;

    struct RenderPassBindindInfo {
        UniquePtr<RenderPassBindingInfo_PSO> pso;
        UniquePtr<RenderPassBindingInfo_Barrier> barrier {nullptr};
    };

    Type_STLUnorderedMap_String<uint32_t> mPassNameToIndex;
    Type_STLVector<RenderPassBindindInfo> mPasses;

    Type_STLUnorderedMap_String<uint32_t> mResNameToIndex;
    Type_STLVector<RenderResource const*> mRenderResources;

    struct Barrier {
        uint32_t resourceIndex;
        vk::ImageLayout layout;
        vk::AccessFlags2 access;
        vk::PipelineStageFlags2 stages;
    };

    struct Barriers {
        Type_STLVector<Barrier> invalidate;
        Type_STLVector<Barrier> flush;
    };

    Type_STLVector<Barriers> mPassBarrierInfos;
};

}  // namespace IntelliDesign_NS::Vulkan::Core