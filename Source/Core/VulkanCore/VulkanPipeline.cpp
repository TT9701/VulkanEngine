#include "VulkanPipeline.hpp"

#include "VulkanContext.hpp"
#include "VulkanShader.hpp"

VulkanPipelineLayout::VulkanPipelineLayout(
    VulkanContext*                                 context,
    vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
    vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
    vk::PipelineLayoutCreateFlags flags, void* pNext)
    : pContext(context),
      mLayout(CreateLayout(setLayouts, pushContants, flags, pNext)) {}

VulkanPipelineLayout::~VulkanPipelineLayout() {
    pContext->GetDeviceHandle().destroy(mLayout);
}

vk::PipelineLayout VulkanPipelineLayout::CreateLayout(
    vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
    vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
    vk::PipelineLayoutCreateFlags flags, void* pNext) const {
    vk::PipelineLayoutCreateInfo info {};
    info.setSetLayouts(setLayouts)
        .setPushConstantRanges(pushContants)
        .setFlags(flags)
        .setPNext(pNext);
    return pContext->GetDeviceHandle().createPipelineLayout(info);
}

VulkanPipeline<PipelineType::Graphics>::VulkanPipeline(
    VulkanContext* context, vk::GraphicsPipelineCreateInfo const& info,
    vk::PipelineCache cache)
    : pContext(context), mPipeline(CreatePipeline(cache, info)) {}

VulkanPipeline<PipelineType::Graphics>::~VulkanPipeline() {
    pContext->GetDeviceHandle().destroy(mPipeline);
}

vk::Pipeline VulkanPipeline<PipelineType::Graphics>::CreatePipeline(
    vk::PipelineCache cache, vk::GraphicsPipelineCreateInfo const& info) const {
    return pContext->GetDeviceHandle()
        .createGraphicsPipeline(cache, info)
        .value;
}

VulkanPipeline<PipelineType::Compute>::VulkanPipeline(
    VulkanContext* context, vk::ComputePipelineCreateInfo const& info,
    vk::PipelineCache cache)
    : pContext(context), mPipeline(CreatePipeline(cache, info)) {}

VulkanPipeline<PipelineType::Compute>::~VulkanPipeline() {
    pContext->GetDeviceHandle().destroy(mPipeline);
}

vk::Pipeline VulkanPipeline<PipelineType::Compute>::CreatePipeline(
    vk::PipelineCache cache, vk::ComputePipelineCreateInfo const& info) const {
    return pContext->GetDeviceHandle().createComputePipeline(cache, info).value;
}

VulkanPipelineBuilder<PipelineType::Graphics>::VulkanPipelineBuilder(
    VulkanPipelineManager* manager)
    : pManager(manager) {
    Clear();
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetLayout(vk::PipelineLayout layout) {
    mPipelineLayout = layout;
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetShaders(::std::span<VulkanShader> shaders) {
    mShaderStages.clear();
    for (const auto& shader : shaders) {
        mShaderStages.push_back(shader.GetStageInfo());
    }
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetInputTopology(vk::PrimitiveTopology topology) {
    mInputAssembly.setPrimitiveRestartEnable(vk::False).setTopology(topology);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetPolygonMode(vk::PolygonMode mode) {
    mRasterizer.setPolygonMode(mode).setLineWidth(1.0f);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>&
VulkanPipelineBuilder<PipelineType::Graphics>::SetCullMode(
    vk::CullModeFlags cullMode, vk::FrontFace frontFace) {
    mRasterizer.setCullMode(cullMode).setFrontFace(frontFace);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>&
VulkanPipelineBuilder<PipelineType::Graphics>::SetMultisampling(
    vk::SampleCountFlagBits sampleCount) {
    mMultisampling
        .setSampleShadingEnable(
            sampleCount == vk::SampleCountFlagBits::e1 ? vk::False : vk::True)
        .setRasterizationSamples(sampleCount)
        .setMinSampleShading(1.0f);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>&
VulkanPipelineBuilder<PipelineType::Graphics>::SetBlending(vk::Bool32 enable) {
    mColorBlendAttachment
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
            | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        .setBlendEnable(enable);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetColorAttachmentFormat(vk::Format format) {
    mColorAttachmentformat = format;
    mRenderInfo.setColorAttachmentCount(1u).setColorAttachmentFormats(
        mColorAttachmentformat);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>& VulkanPipelineBuilder<
    PipelineType::Graphics>::SetDepthStencilFormat(vk::Format format) {
    mRenderInfo.setDepthAttachmentFormat(format);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>&
VulkanPipelineBuilder<PipelineType::Graphics>::SetDepth(vk::Bool32 depthTest,
                                                        vk::Bool32 depthWrite,
                                                        vk::CompareOp compare) {
    mDepthStencil.setDepthTestEnable(depthTest)
        .setDepthWriteEnable(depthWrite)
        .setDepthCompareOp(compare)
        .setDepthBoundsTestEnable(vk::False);
    return *this;
}

VulkanPipelineBuilder<PipelineType::Graphics>&
VulkanPipelineBuilder<PipelineType::Graphics>::SetStencil(vk::Bool32 stencil) {
    mDepthStencil.setStencilTestEnable(stencil);
    return *this;
}

SharedPtr<VulkanPipeline<PipelineType::Graphics>>
VulkanPipelineBuilder<PipelineType::Graphics>::Build(std::string const& name,
                                                     vk::PipelineCache  cache,
                                                     void*              pNext) {
    vk::PipelineViewportStateCreateInfo viewportState {};
    viewportState.setViewportCount(1u).setScissorCount(1u);

    vk::PipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.setLogicOpEnable(vk::False)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachments(mColorBlendAttachment);

    vk::PipelineVertexInputStateCreateInfo vertexInput {};

    ::std::array dynamicStates = {vk::DynamicState::eViewport,
                                  vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicInfo {};
    dynamicInfo.setDynamicStates(dynamicStates);

    vk::GraphicsPipelineCreateInfo createInfo {};

    mRenderInfo.setPNext(pNext);
    createInfo.setPNext(&mRenderInfo)
        .setStages(mShaderStages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&mInputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&mRasterizer)
        .setPMultisampleState(&mMultisampling)
        .setPColorBlendState(&colorBlending)
        .setPDepthStencilState(&mDepthStencil)
        .setLayout(mPipelineLayout)
        .setPDynamicState(&dynamicInfo);

    auto ptr = MakeShared<VulkanPipeline<PipelineType::Graphics>>(
        pManager->pContext, createInfo, cache);

    pManager->mGraphicsPipelines.emplace(name, ptr);

    Clear();

    return ptr;
}

void VulkanPipelineBuilder<PipelineType::Graphics>::Clear() {
    mShaderStages.clear();
    mPipelineLayout        = vk::PipelineLayout {};
    mInputAssembly         = vk::PipelineInputAssemblyStateCreateInfo {};
    mRasterizer            = vk::PipelineRasterizationStateCreateInfo {};
    mColorBlendAttachment  = vk::PipelineColorBlendAttachmentState {};
    mMultisampling         = vk::PipelineMultisampleStateCreateInfo {};
    mDepthStencil          = vk::PipelineDepthStencilStateCreateInfo {};
    mRenderInfo            = vk::PipelineRenderingCreateInfo {};
    mColorAttachmentformat = vk::Format {};
}

VulkanPipelineBuilder<PipelineType::Compute>::VulkanPipelineBuilder(
    VulkanPipelineManager* manager)
    : pManager(manager) {
    Clear();
}

VulkanPipelineBuilder<PipelineType::Compute>& VulkanPipelineBuilder<
    PipelineType::Compute>::SetShader(VulkanShader const& shader) {
    mStageInfo = shader.GetStageInfo();
    return *this;
}

VulkanPipelineBuilder<PipelineType::Compute>& VulkanPipelineBuilder<
    PipelineType::Compute>::SetLayout(vk::PipelineLayout pipelineLayout) {
    mPipelineLayout = pipelineLayout;
    return *this;
}

VulkanPipelineBuilder<PipelineType::Compute>& VulkanPipelineBuilder<
    PipelineType::Compute>::SetFlags(vk::PipelineCreateFlags flags) {
    mFlags = flags;
    return *this;
}

VulkanPipelineBuilder<PipelineType::Compute>& VulkanPipelineBuilder<
    PipelineType::Compute>::SetBaseHandle(vk::Pipeline baseHandle) {
    mBaseHandle = baseHandle;
    return *this;
}

VulkanPipelineBuilder<PipelineType::Compute>&
VulkanPipelineBuilder<PipelineType::Compute>::SetBaseIndex(int32_t index) {
    mBaseIndex = index;
    return *this;
}

SharedPtr<VulkanPipeline<PipelineType::Compute>>
VulkanPipelineBuilder<PipelineType::Compute>::Build(::std::string const& name,
                                                    vk::PipelineCache    cache,
                                                    void* pNext) {
    vk::ComputePipelineCreateInfo info {};
    info.setFlags(mFlags)
        .setLayout(mPipelineLayout)
        .setStage(mStageInfo)
        .setBasePipelineHandle(mBaseHandle)
        .setBasePipelineIndex(mBaseIndex)
        .setPNext(pNext);

    Clear();

    auto ptr = MakeShared<VulkanPipeline<PipelineType::Compute>>(
        pManager->pContext, info, cache);

    pManager->mComputePipelines.emplace(name, ptr);

    return ptr;
}

void VulkanPipelineBuilder<PipelineType::Compute>::Clear() {
    mStageInfo      = vk::PipelineShaderStageCreateInfo {};
    mPipelineLayout = vk::PipelineLayout {};
    mFlags          = vk::PipelineCreateFlags {};
    mBaseHandle     = vk::Pipeline {};
    mBaseIndex      = int32_t {};
}

VulkanPipelineManager::VulkanPipelineManager(VulkanContext* contex)
    : pContext(contex),
      mComputePipelineBuilder {this},
      mGraphicsPipelineBuilder {this} {}

SharedPtr<VulkanPipelineLayout> VulkanPipelineManager::CreateLayout(
    ::std::string const&                           name,
    vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
    vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
    vk::PipelineLayoutCreateFlags flags, void* pNext) {
    const auto ptr = MakeShared<VulkanPipelineLayout>(
        pContext, setLayouts, pushContants, flags, pNext);

    mPipelineLayouts.emplace(name, ptr);

    return ptr;
}

vk::PipelineLayout VulkanPipelineManager::GetLayoutHandle(
    std::string const& name) const {
    return mPipelineLayouts.at(name)->GetHandle();
}

vk::Pipeline VulkanPipelineManager::GetComputePipeline(
    std::string const& name) const {
    return mComputePipelines.at(name)->GetHandle();
}

vk::Pipeline VulkanPipelineManager::GetGraphicsPipeline(
    std::string const& name) const {
    return mGraphicsPipelines.at(name)->GetHandle();
}

VulkanPipelineManager::CPBuilder&
VulkanPipelineManager::GetComputePipelineBuilder() {
    return mComputePipelineBuilder;
}

VulkanPipelineManager::GPBuilder&
VulkanPipelineManager::GetGraphicsPipelineBuilder() {
    return mGraphicsPipelineBuilder;
}