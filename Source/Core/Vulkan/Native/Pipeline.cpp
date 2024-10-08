#include "Pipeline.hpp"

#include "Core/Vulkan/Manager/Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineLayout::PipelineLayout(Context* context, ShaderStats const& stats,
                               vk::PipelineLayoutCreateFlags flags, void* pNext)
    : pContext(context),
      mPushContantRanges(stats.pushContant),
      mLayout(CreateLayout(stats, flags, pNext)) {}

PipelineLayout::~PipelineLayout() {
    pContext->GetDeviceHandle().destroy(mLayout);
}

Type_STLVector<vk::PushConstantRange> const& PipelineLayout::GetPushConstants()
    const {
    return mPushContantRanges;
}

vk::PipelineLayout PipelineLayout::CreateLayout(
    ShaderStats stats, vk::PipelineLayoutCreateFlags flags, void* pNext) const {
    vk::PipelineLayoutCreateInfo info {};
    info.setSetLayouts(stats.descSetLayouts)
        .setPushConstantRanges(mPushContantRanges)
        .setFlags(flags)
        .setPNext(pNext);
    return pContext->GetDeviceHandle().createPipelineLayout(info);
}

Pipeline<PipelineType::Graphics>::Pipeline(
    Context* context, vk::GraphicsPipelineCreateInfo const& info,
    vk::PipelineCache cache)
    : pContext(context), mPipeline(CreatePipeline(cache, info)) {}

Pipeline<PipelineType::Graphics>::~Pipeline() {
    pContext->GetDeviceHandle().destroy(mPipeline);
}

vk::Pipeline Pipeline<PipelineType::Graphics>::CreatePipeline(
    vk::PipelineCache cache, vk::GraphicsPipelineCreateInfo const& info) const {
    return pContext->GetDeviceHandle()
        .createGraphicsPipeline(cache, info)
        .value;
}

Pipeline<PipelineType::Compute>::Pipeline(
    Context* context, vk::ComputePipelineCreateInfo const& info,
    vk::PipelineCache cache)
    : pContext(context), mPipeline(CreatePipeline(cache, info)) {}

Pipeline<PipelineType::Compute>::~Pipeline() {
    pContext->GetDeviceHandle().destroy(mPipeline);
}

vk::Pipeline Pipeline<PipelineType::Compute>::CreatePipeline(
    vk::PipelineCache cache, vk::ComputePipelineCreateInfo const& info) const {
    return pContext->GetDeviceHandle().createComputePipeline(cache, info).value;
}

}  // namespace IntelliDesign_NS::Vulkan::Core