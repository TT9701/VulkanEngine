#include "VulkanPipeline.hpp"

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetLayout(
    vk::PipelineLayout layout) {
    mPipelineLayout = layout;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetShaders(
    vk::ShaderModule vertexShader, vk::ShaderModule fragmentShader) {
    mShaderStages.clear();

    mShaderStages.push_back(
        {{}, vk::ShaderStageFlagBits::eVertex, vertexShader, "main"});

    mShaderStages.push_back(
        {{}, vk::ShaderStageFlagBits::eFragment, fragmentShader, "main"});

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetInputTopology(
    vk::PrimitiveTopology topology) {
    mInputAssembly.setPrimitiveRestartEnable(vk::False).setTopology(topology);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetPolygonMode(
    vk::PolygonMode mode) {
    mRasterizer.setPolygonMode(mode).setLineWidth(1.0f);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetCullMode(
    vk::CullModeFlags cullMode, vk::FrontFace frontFace) {
    mRasterizer.setCullMode(cullMode).setFrontFace(frontFace);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetMultisampling(
    vk::SampleCountFlagBits sampleCount) {
    mMultisampling
        .setSampleShadingEnable(
            sampleCount == vk::SampleCountFlagBits::e1 ? vk::False : vk::True)
        .setRasterizationSamples(sampleCount)
        .setMinSampleShading(1.0f);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetBlending(
    vk::Bool32 enable) {
    mColorBlendAttachment
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
            | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        .setBlendEnable(enable);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetColorAttachmentFormat(
    vk::Format format) {
    mColorAttachmentformat = format;

    mRenderInfo.setColorAttachmentCount(1u).setColorAttachmentFormats(
        mColorAttachmentformat);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetDepthStencilFormat(
    vk::Format format) {
    mRenderInfo.setDepthAttachmentFormat(format);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetDepth(
    vk::Bool32 depthTest, vk::Bool32 depthWrite, vk::CompareOp compare) {
    mDepthStencil.setDepthTestEnable(depthTest)
        .setDepthWriteEnable(depthWrite)
        .setDepthCompareOp(compare)
        .setDepthBoundsTestEnable(vk::False);

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::SetStencil(
    vk::Bool32 stencil) {
    mDepthStencil.setStencilTestEnable(stencil);

    return *this;
}

void GraphicsPipelineBuilder::Clear() {
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

vk::Pipeline GraphicsPipelineBuilder::Build(vk::Device device) {
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

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo {};
    pipelineCreateInfo.setPNext(&mRenderInfo)
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

    return device.createGraphicsPipeline({}, pipelineCreateInfo).value;
}