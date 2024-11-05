#include "Pipeline.hpp"

#include "Core/Vulkan/Manager/Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineLayout::PipelineLayout(Context* context, ShaderProgram* program,
                               vk::PipelineLayoutCreateFlags flags, void* pNext)
    : pContext(context),
      pProgram(program),
      mLayout(CreateLayout(flags, pNext)) {
    int i = 0;
}

PipelineLayout::~PipelineLayout() {
    pContext->GetDeviceHandle().destroy(mLayout);
}

Type_STLVector<DescriptorSetLayout*> PipelineLayout::GetDescSetLayoutDatas()
    const {
    return pProgram->GetCombinedDescLayouts();
}

Type_STLVector<vk::PushConstantRange> PipelineLayout::GetPCRanges() const {
    return pProgram->GetPCRanges();
}

ShaderProgram::Type_CombinedPushContant const&
PipelineLayout::GetCombinedPushContant() const {
    return pProgram->mCombinedPushContants;
}

Type_STLVector<Type_STLString> const& PipelineLayout::GetRTVNames() const {
    return pProgram->mRtvNames;
}

vk::PipelineLayout PipelineLayout::CreateLayout(
    vk::PipelineLayoutCreateFlags flags, void* pNext) const {
    vk::PipelineLayoutCreateInfo info {};
    auto layouts = pProgram->GetCombinedDescLayoutHandles();
    auto ranges = pProgram->GetPCRanges();
    info.setSetLayouts(layouts)
        .setPushConstantRanges(ranges)
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