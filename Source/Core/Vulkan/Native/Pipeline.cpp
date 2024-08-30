#include "Pipeline.hpp"

#include "Core/Vulkan/Manager/Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineLayout::PipelineLayout(Context* context,
                               ::std::span<vk::DescriptorSetLayout> setLayouts,
                               ::std::span<vk::PushConstantRange> pushContants,
                               vk::PipelineLayoutCreateFlags flags, void* pNext)
    : pContext(context),
      mLayout(CreateLayout(setLayouts, pushContants, flags, pNext)) {}

PipelineLayout::~PipelineLayout() {
    pContext->GetDeviceHandle().destroy(mLayout);
}

vk::PipelineLayout PipelineLayout::CreateLayout(
    ::std::span<vk::DescriptorSetLayout> setLayouts,
    ::std::span<vk::PushConstantRange> pushContants,
    vk::PipelineLayoutCreateFlags flags, void* pNext) const {
    vk::PipelineLayoutCreateInfo info {};
    info.setSetLayouts(setLayouts)
        .setPushConstantRanges(pushContants)
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