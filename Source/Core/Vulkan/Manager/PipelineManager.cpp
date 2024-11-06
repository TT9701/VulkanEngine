#include "PipelineManager.h"

#include <spirv_glsl.hpp>

#include "Context.h"
#include "Core/Vulkan/Native/Shader.h"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineBuilder<PipelineType::Graphics>::PipelineBuilder(
    PipelineManager* manager)
    : pManager(manager) {
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

SharedPtr<Pipeline<PipelineType::Graphics>>
PipelineBuilder<PipelineType::Graphics>::Build(const char* name,
                                               vk::PipelineCache cache,
                                               void* pNext) {
    auto pipelineName = pManager->ParsePipelineName(name);
    auto pipelineLayoutName = pManager->ParsePipelineLayoutName(name);

    auto pipelineLayout = pManager->CreateLayout(
        pipelineLayoutName.c_str(), pProgram);

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

    auto pipeline = MakeShared<Pipeline<PipelineType::Graphics>>(
        pManager->pContext, createInfo, cache);

    auto shaders = pProgram->GetShaderArray();
    for (const auto& modules : shaders) {
        if (modules)
            modules->GetMutex().unlock();
    }
    pManager->pContext->SetName(pipeline->GetHandle(), pipelineName);
    pManager->mGraphicsPipelines.emplace(pipelineName, pipeline);

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
    PipelineManager* manager)
    : pManager(manager) {
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

SharedPtr<Pipeline<PipelineType::Compute>>
PipelineBuilder<PipelineType::Compute>::Build(const char* name,
                                              vk::PipelineCache cache,
                                              void* pNext) {
    auto pipelineName = pManager->ParsePipelineName(name);
    auto pipelineLayoutName = pManager->ParsePipelineLayoutName(name);

    auto pipelineLayout =
        pManager->CreateLayout(pipelineLayoutName.c_str(), pProgram);

    vk::ComputePipelineCreateInfo info {};
    info.setFlags(mFlags)
        .setLayout(pipelineLayout->GetHandle())
        .setStage(mStageInfo)
        .setBasePipelineHandle(mBaseHandle)
        .setBasePipelineIndex(mBaseIndex)
        .setPNext(pNext);

    Clear();

    auto pipeline = MakeShared<Pipeline<PipelineType::Compute>>(
        pManager->pContext, info, cache);

    (*pProgram)[ShaderStage::Compute]->GetMutex().unlock();
    pManager->pContext->SetName(pipeline->GetHandle(), pipelineName);
    pManager->mComputePipelines.emplace(pipelineName.c_str(), pipeline);

    return pipeline;
}

void PipelineBuilder<PipelineType::Compute>::Clear() {
    mStageInfo = vk::PipelineShaderStageCreateInfo {};
    mPipelineLayout = vk::PipelineLayout {};
    mFlags = vk::PipelineCreateFlags {};
    mBaseHandle = vk::Pipeline {};
    mBaseIndex = int32_t {};
}

PipelineManager::PipelineManager(Context* contex) : pContext(contex) {}

SharedPtr<PipelineLayout> PipelineManager::CreateLayout(
    const char* name, ShaderProgram* program,
    vk::PipelineLayoutCreateFlags flags, void* pNext) {
    const auto ptr =
        MakeShared<PipelineLayout>(pContext, program, flags, pNext);

    pContext->SetName(ptr->GetHandle(), name);
    mPipelineLayouts.emplace(name, ptr);

    return ptr;
}

vk::PipelineLayout PipelineManager::GetLayoutHandle(const char* name) const {
    return GetLayout(name)->GetHandle();
}

PipelineLayout* PipelineManager::GetLayout(const char* name) const {
    return mPipelineLayouts.at(ParsePipelineLayoutName(name)).get();
}

vk::Pipeline PipelineManager::GetComputePipelineHandle(const char* name) const {

    return mComputePipelines.at(ParsePipelineName(name))->GetHandle();
}

vk::Pipeline PipelineManager::GetGraphicsPipelineHandle(
    const char* name) const {
    return mGraphicsPipelines.at(ParsePipelineName(name))->GetHandle();
}

PipelineManager::Type_ComputePipelines const&
PipelineManager::GetComputePipelines() const {
    return mComputePipelines;
}

PipelineManager::Type_GraphicsPipelines const&
PipelineManager::GetGraphicsPipelines() const {
    return mGraphicsPipelines;
}

PipelineManager::Type_CPBuilder PipelineManager::GetComputePipelineBuilder() {
    return Type_CPBuilder {this};
}

PipelineManager::Type_GPBuilder PipelineManager::GetGraphicsPipelineBuilder() {
    return Type_GPBuilder {this};
}

void PipelineManager::BindComputePipeline(vk::CommandBuffer cmd,
                                          const char* name) {

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute,
                     GetComputePipelineHandle(name));
}

void PipelineManager::BindGraphicsPipeline(vk::CommandBuffer cmd,
                                           const char* name) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                     GetGraphicsPipelineHandle(name));
}

Type_STLString PipelineManager::ParsePipelineName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName};
}

Type_STLString PipelineManager::ParsePipelineLayoutName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName};
}

}  // namespace IntelliDesign_NS::Vulkan::Core