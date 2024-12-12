#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Shader.h"

namespace IntelliDesign_NS::Vulkan::Core {

enum class PipelineType { Graphics, Compute };

class Context;

class PipelineLayout {
public:
    PipelineLayout(VulkanContext* context, ShaderProgram* program,
                   vk::PipelineLayoutCreateFlags flags = {},
                   void* pNext = nullptr);
    ~PipelineLayout();
    CLASS_MOVABLE_ONLY(PipelineLayout);

public:
    vk::PipelineLayout GetHandle() const { return mLayout; }

    Type_STLVector<DescriptorSetLayout*> GetDescSetLayoutDatas() const;

    Type_STLVector<vk::PushConstantRange> GetPCRanges() const;

    ShaderProgram::Type_CombinedPushContant const& GetCombinedPushContant()
        const;

    Type_STLVector<Type_STLString> const& GetRTVNames() const;

private:
    vk::PipelineLayout CreateLayout(vk::PipelineLayoutCreateFlags flags,
                                    void* pNext) const;

private:
    VulkanContext* pContext;
    ShaderProgram* pProgram;

    vk::PipelineLayout mLayout;
};

template <PipelineType Type>
class Pipeline;

template <>
class Pipeline<PipelineType::Graphics> {
public:
    Pipeline(VulkanContext* context, vk::GraphicsPipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    ~Pipeline();
    CLASS_MOVABLE_ONLY(Pipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache cache,
        vk::GraphicsPipelineCreateInfo const& info) const;

private:
    VulkanContext* pContext;

    vk::Pipeline mPipeline;
};

template <>
class Pipeline<PipelineType::Compute> {
public:
    Pipeline(VulkanContext* context, vk::ComputePipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    ~Pipeline();
    CLASS_MOVABLE_ONLY(Pipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache cache,
        vk::ComputePipelineCreateInfo const& info) const;

private:
    VulkanContext* pContext;

    vk::Pipeline mPipeline;
};

}  // namespace IntelliDesign_NS::Vulkan::Core