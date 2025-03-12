#pragma once

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Pipeline.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Shader;
class PipelineManager;

/**
 * Use PipelineBuilder to build Pipelines
 */
class PipelineManager {
    using Type_CPBuilder = PipelineBuilder<PipelineType::Compute>;
    using Type_GPBuilder = PipelineBuilder<PipelineType::Graphics>;

    using Type_PipelineLayouts =
        Type_STLUnorderedMap_String<SharedPtr<PipelineLayout>>;
    using Type_Pipelines = Type_STLUnorderedMap_String<SharedPtr<Pipeline>>;

public:
    PipelineManager(VulkanContext& contex);
    ~PipelineManager() = default;
    CLASS_MOVABLE_ONLY(PipelineManager);

    friend Type_CPBuilder;
    friend Type_GPBuilder;

public:
    PipelineLayout* CreateLayout(const char* name, ShaderProgram* program,
                                 vk::PipelineLayoutCreateFlags flags = {},
                                 void* pNext = nullptr);

    vk::PipelineLayout GetLayoutHandle(const char* name) const;
    PipelineLayout* GetLayout(const char* name) const;

    vk::Pipeline GetPipelineHandle(const char* name) const;
    Pipeline& GetPipeline(const char* name) const;

    Type_Pipelines const& GetPipelines() const;

    Type_CPBuilder GetBuilder_Compute();
    Type_GPBuilder GetBuilder_Graphics();

    void BindPipeline(vk::CommandBuffer cmd, const char* name);

private:
    Type_STLString ParsePipelineName(const char* pipelineName) const;
    Type_STLString ParsePipelineLayoutName(const char* pipelineName) const;

private:
    VulkanContext& mContext;

    Type_PipelineLayouts mPipelineLayouts;
    Type_Pipelines mPipelines;
};

}  // namespace IntelliDesign_NS::Vulkan::Core