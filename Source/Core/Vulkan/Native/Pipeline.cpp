#include "Pipeline.h"

#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineLayout::PipelineLayout(VulkanContext& context, ShaderProgram& program,
                               vk::PipelineLayoutCreateFlags flags, void* pNext)
    : mContext(context),
      mProgram(program),
      mLayout(CreateLayout(flags, pNext)) {
    int i = 0;
}

PipelineLayout::~PipelineLayout() {
    mContext.GetDevice()->destroy(mLayout);
}

Type_STLVector<DescriptorSetLayout*> PipelineLayout::GetDescSetLayoutDatas()
    const {
    return mProgram.GetCombinedDescLayouts();
}

Type_STLVector<vk::PushConstantRange> PipelineLayout::GetPCRanges() const {
    return mProgram.GetPCRanges();
}

ShaderProgram::Type_CombinedPushContant const&
PipelineLayout::GetCombinedPushContant() const {
    return mProgram.mCombinedPushContants;
}

Type_STLVector<Type_STLString> const& PipelineLayout::GetRTVNames() const {
    return mProgram.mRtvNames;
}

vk::PipelineLayout PipelineLayout::CreateLayout(
    vk::PipelineLayoutCreateFlags flags, void* pNext) const {
    vk::PipelineLayoutCreateInfo info {};
    auto layouts = mProgram.GetCombinedDescLayoutHandles();
    auto ranges = mProgram.GetPCRanges();
    info.setSetLayouts(layouts)
        .setPushConstantRanges(ranges)
        .setFlags(flags)
        .setPNext(pNext);
    return mContext.GetDevice()->createPipelineLayout(info);
}

Pipeline::Pipeline(VulkanContext& context,
                   vk::GraphicsPipelineCreateInfo const& info,
                   vk::PipelineCache cache)
    : mContext(context), mType(PipelineType::Graphics) {
    mHandle = mContext.GetDevice()->createGraphicsPipeline(cache, info).value;
}

Pipeline::Pipeline(VulkanContext& context,
                   vk::ComputePipelineCreateInfo const& info,
                   vk::PipelineCache cache)
    : mContext(context), mType(PipelineType::Compute) {
    mHandle = mContext.GetDevice()->createComputePipeline(cache, info).value;
}

Pipeline::~Pipeline() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::Pipeline Pipeline::GetHandle() const {
    return mHandle;
}

PipelineType Pipeline::GetType() const {
    return mType;
}

PipelineBuilder<PipelineType::Graphics>::PipelineBuilder(
    PipelineManager& manager)
    : mManager(manager) {
    Clear();
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetLayout(vk::PipelineLayout layout) {
    mPipelineLayout = layout;
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetShaderProgram(ShaderProgram* program) {
    mShaderStages.clear();
    pProgram = program;
    for (uint32_t i = 0; i < Utils::EnumCast(ShaderStage::Count); ++i) {
        if (auto shader = (*pProgram)[static_cast<ShaderStage>(i)]) {
            shader->GetMutex().lock();
            mShaderStages.push_back(shader->GetStageInfo());
        }
    }
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetInputTopology(vk::PrimitiveTopology topology) {
    mInputAssembly.setPrimitiveRestartEnable(vk::False).setTopology(topology);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetPolygonMode(vk::PolygonMode mode) {
    mRasterizer.setPolygonMode(mode).setLineWidth(1.0f);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetCullMode(vk::CullModeFlags cullMode,
                                                     vk::FrontFace frontFace) {
    mRasterizer.setCullMode(cullMode).setFrontFace(frontFace);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetMultisampling(
    vk::SampleCountFlagBits sampleCount) {
    mMultisampling
        .setSampleShadingEnable(
            sampleCount == vk::SampleCountFlagBits::e1 ? vk::False : vk::True)
        .setRasterizationSamples(sampleCount)
        .setMinSampleShading(1.0f);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetBlending(vk::Bool32 enable) {
    mColorBlendAttachment
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
            | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        .setBlendEnable(enable);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetColorAttachmentFormat(vk::Format format) {
    mColorAttachmentformat = format;
    mRenderInfo.setColorAttachmentCount(1u).setColorAttachmentFormats(
        mColorAttachmentformat);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetDepthStencilFormat(vk::Format format) {
    mRenderInfo.setDepthAttachmentFormat(format);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetDepth(vk::Bool32 depthTest,
                                                  vk::Bool32 depthWrite,
                                                  vk::CompareOp compare) {
    mDepthStencil.setDepthTestEnable(depthTest)
        .setDepthWriteEnable(depthWrite)
        .setDepthCompareOp(compare)
        .setDepthBoundsTestEnable(vk::False);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetStencil(vk::Bool32 stencil) {
    mDepthStencil.setStencilTestEnable(stencil);
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetBaseHandle(vk::Pipeline baseHandle) {
    mBaseHandle = baseHandle;
    return *this;
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetBaseIndex(int32_t index) {
    mBaseIndex = index;
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetFlags(vk::PipelineCreateFlags flags) {
    mFlags = flags;
    return *this;
}

SharedPtr<Pipeline> PipelineBuilder<PipelineType::Graphics>::Build(
    const char* name, vk::PipelineCache cache, void* pNext) {
    auto pipelineName = mManager.ParsePipelineName(name);
    auto pipelineLayoutName = mManager.ParsePipelineLayoutName(name);

    auto pipelineLayout =
        mManager.CreateLayout(pipelineLayoutName.c_str(), pProgram);

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
        .setLayout(pipelineLayout->GetHandle())
        .setPDynamicState(&dynamicInfo)
        .setBasePipelineHandle(mBaseHandle)
        .setBasePipelineIndex(mBaseIndex)
        .setFlags(mFlags);

    auto pipeline = MakeShared<Pipeline>(mManager.mContext, createInfo, cache);

    auto shaders = pProgram->GetShaderArray();
    for (const auto& modules : shaders) {
        if (modules)
            modules->GetMutex().unlock();
    }
    mManager.mContext.SetName(pipeline->GetHandle(), pipelineName);
    mManager.mPipelines.emplace(pipelineName, pipeline);

    Clear();

    return pipeline;
}

void PipelineBuilder<PipelineType::Graphics>::Clear() {
    mShaderStages.clear();
    pProgram = nullptr;
    mPipelineLayout = vk::PipelineLayout {};
    mInputAssembly = vk::PipelineInputAssemblyStateCreateInfo {};
    mRasterizer = vk::PipelineRasterizationStateCreateInfo {};
    mColorBlendAttachment = vk::PipelineColorBlendAttachmentState {};
    mMultisampling = vk::PipelineMultisampleStateCreateInfo {};
    mDepthStencil = vk::PipelineDepthStencilStateCreateInfo {};
    mRenderInfo = vk::PipelineRenderingCreateInfo {};
    mColorAttachmentformat = vk::Format {};
    mFlags = {};
}

PipelineBuilder<PipelineType::Compute>::PipelineBuilder(
    PipelineManager& manager)
    : mManager(manager) {
    Clear();
}

PipelineBuilder<PipelineType::Compute>& PipelineBuilder<
    PipelineType::Compute>::SetShaderProgram(ShaderProgram* program) {
    pProgram = program;
    auto shader = (*pProgram)[ShaderStage::Compute];
    shader->GetMutex().lock();
    mStageInfo = shader->GetStageInfo();
    return *this;
}

PipelineBuilder<PipelineType::Compute>& PipelineBuilder<
    PipelineType::Compute>::SetLayout(vk::PipelineLayout pipelineLayout) {
    mPipelineLayout = pipelineLayout;
    return *this;
}

PipelineBuilder<PipelineType::Compute>& PipelineBuilder<
    PipelineType::Compute>::SetFlags(vk::PipelineCreateFlags flags) {
    mFlags = flags;
    return *this;
}

PipelineBuilder<PipelineType::Compute>&
PipelineBuilder<PipelineType::Compute>::SetBaseHandle(vk::Pipeline baseHandle) {
    mBaseHandle = baseHandle;
    return *this;
}

PipelineBuilder<PipelineType::Compute>&
PipelineBuilder<PipelineType::Compute>::SetBaseIndex(int32_t index) {
    mBaseIndex = index;
    return *this;
}

SharedPtr<Pipeline> PipelineBuilder<PipelineType::Compute>::Build(
    const char* name, vk::PipelineCache cache, void* pNext) {
    auto pipelineName = mManager.ParsePipelineName(name);
    auto pipelineLayoutName = mManager.ParsePipelineLayoutName(name);

    auto pipelineLayout =
        mManager.CreateLayout(pipelineLayoutName.c_str(), pProgram);

    vk::ComputePipelineCreateInfo info {};
    info.setFlags(mFlags)
        .setLayout(pipelineLayout->GetHandle())
        .setStage(mStageInfo)
        .setBasePipelineHandle(mBaseHandle)
        .setBasePipelineIndex(mBaseIndex)
        .setPNext(pNext);

    Clear();

    auto pipeline = MakeShared<Pipeline>(mManager.mContext, info, cache);

    (*pProgram)[ShaderStage::Compute]->GetMutex().unlock();
    mManager.mContext.SetName(pipeline->GetHandle(), pipelineName);
    mManager.mPipelines.emplace(pipelineName.c_str(), pipeline);

    return pipeline;
}

void PipelineBuilder<PipelineType::Compute>::Clear() {
    mStageInfo = vk::PipelineShaderStageCreateInfo {};
    mPipelineLayout = vk::PipelineLayout {};
    mFlags = vk::PipelineCreateFlags {};
    mBaseHandle = vk::Pipeline {};
    mBaseIndex = int32_t {};
}

}  // namespace IntelliDesign_NS::Vulkan::Core