#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Native/Shader.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

enum class PipelineType { Graphics, Compute };

class Context;

struct ShaderStats {
    Type_STLVector<vk::DescriptorSetLayout> descSetLayouts;
    Type_STLVector<vk::PushConstantRange> pushContant;
};

class PipelineLayout {
public:
    PipelineLayout(
        Context* context,
        Type_STLVector<DescriptorSetLayout*> const& descSetLayoutDatas,
        ShaderStats const& stats, vk::PipelineLayoutCreateFlags flags = {},
        void* pNext = nullptr);
    ~PipelineLayout();
    MOVABLE_ONLY(PipelineLayout);

public:
    vk::PipelineLayout GetHandle() const { return mLayout; }

    Type_STLVector<DescriptorSetLayout*> const& GetDescSetLayoutDatas()
        const;

    Type_STLVector<vk::PushConstantRange> const& GetPushConstants() const;

private:
    vk::PipelineLayout CreateLayout(ShaderStats stats,
                                    vk::PipelineLayoutCreateFlags flags,
                                    void* pNext) const;

private:
    Context* pContext;

    Type_STLVector<DescriptorSetLayout*> mDescSetLayoutDatas;
    Type_STLVector<vk::PushConstantRange> mPushContantRanges;
    vk::PipelineLayout mLayout;
};

template <PipelineType Type>
class Pipeline;

template <>
class Pipeline<PipelineType::Graphics> {
public:
    Pipeline(Context* context, vk::GraphicsPipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    ~Pipeline();
    MOVABLE_ONLY(Pipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache cache,
        vk::GraphicsPipelineCreateInfo const& info) const;

private:
    Context* pContext;

    vk::Pipeline mPipeline;
};

template <>
class Pipeline<PipelineType::Compute> {
public:
    Pipeline(Context* context, vk::ComputePipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    ~Pipeline();
    MOVABLE_ONLY(Pipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache cache,
        vk::ComputePipelineCreateInfo const& info) const;

private:
    Context* pContext;

    vk::Pipeline mPipeline;
};

}  // namespace IntelliDesign_NS::Vulkan::Core