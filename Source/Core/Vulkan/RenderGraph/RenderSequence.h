#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

namespace IntelliDesign_NS::Vulkan::Core {
class IRenderPassBindingInfo;

enum class RenderQueueType { Graphics, Compute, AsyncCompute };

class RenderSequenceConfig {
public:
};

class VulkanContext;
class RenderResourceManager;
class PipelineManager;
class Swapchain;
class RenderPassBindingInfo_PSO;
class RenderPassBindingInfo_Barrier;

class RenderSequence {
    struct RenderPassBindingInfo {
        RenderSequence& sequence;
        UniquePtr<RenderPassBindingInfo_PSO> pso;
        UniquePtr<RenderPassBindingInfo_Barrier> preBarrieres {nullptr};
        UniquePtr<RenderPassBindingInfo_Barrier> postBarrieres {nullptr};

        void RecordCmd(vk::CommandBuffer cmd);
        void Update(Type_STLVector<Type_STLString> const& resNames);
        void OnResize(vk::Extent2D extent);
    };

public:
    RenderSequence(VulkanContext& context, RenderResourceManager& resMgr,
                   PipelineManager& pipelineMgr, DescriptorSetPool& descPool);

    ~RenderSequence() = default;

    CLASS_NO_COPY_MOVE(RenderSequence);

    RenderPassBindingInfo& AddRenderPass(const char* name,
                                         RenderQueueType type);

    RenderPassBindingInfo& FindRenderPass(const char* name);

    RenderPassBindingInfo& GetRenderToSwapchainPass();

    void RecordPass(const char* name, vk::CommandBuffer cmd);

    void GenerateBarriers();

    friend RenderPassBindingInfo_PSO;

private:
    uint32_t AddRenderResource(const char* name);

    VulkanContext& mContext;
    RenderResourceManager& mResMgr;
    PipelineManager& mPipelineMgr;
    DescriptorSetPool& mDescPool;

    Type_STLUnorderedMap_String<uint32_t> mPassNameToIndex;
    Type_STLVector<RenderPassBindingInfo> mPasses;

    Type_STLUnorderedMap_String<uint32_t> mResNameToIndex;
    Type_STLVector<RenderResource const*> mRenderResources;

    struct Barrier {
        uint32_t resourceIndex;
        vk::ImageLayout layout;
        vk::AccessFlags2 access;
        vk::PipelineStageFlags2 stages;
    };

    Type_STLVector<Type_STLVector<Barrier>> mPassBarrierInfos;
};

}  // namespace IntelliDesign_NS::Vulkan::Core