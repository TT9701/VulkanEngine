#pragma once

#include <Core/Utilities/VulkanUtilities.hpp>

enum class PipelineType { Graphics, Compute };

class PipelineManager {
public:

private:
    // ::std::unordered_map<::std::string, vk::Pipeline> 
};

class GraphicsPipelineBuilder {
public:
    ::std::vector<vk::PipelineShaderStageCreateInfo> mShaderStages {};

    vk::PipelineInputAssemblyStateCreateInfo mInputAssembly;
    vk::PipelineRasterizationStateCreateInfo mRasterizer;
    vk::PipelineColorBlendAttachmentState    mColorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo   mMultisampling;
    vk::PipelineLayout                       mPipelineLayout;
    vk::PipelineDepthStencilStateCreateInfo  mDepthStencil;
    vk::PipelineRenderingCreateInfo          mRenderInfo;
    vk::Format                               mColorAttachmentformat;

    GraphicsPipelineBuilder() { Clear(); }

    GraphicsPipelineBuilder& SetLayout(vk::PipelineLayout layout);

    GraphicsPipelineBuilder& SetShaders(vk::ShaderModule vertexShader,
                                        vk::ShaderModule fragmentShader);

    GraphicsPipelineBuilder& SetInputTopology(vk::PrimitiveTopology topology);

    GraphicsPipelineBuilder& SetPolygonMode(vk::PolygonMode mode);

    GraphicsPipelineBuilder& SetCullMode(vk::CullModeFlags cullMode,
                                         vk::FrontFace     frontFace);

    // TODO: finish multisampling - pipeline settings
    GraphicsPipelineBuilder& SetMultisampling(
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    // TODO: finish blending - pipeline settings
    GraphicsPipelineBuilder& SetBlending(vk::Bool32 enable = vk::False);

    GraphicsPipelineBuilder& SetColorAttachmentFormat(vk::Format format);

    GraphicsPipelineBuilder& SetDepthStencilFormat(vk::Format format);

    GraphicsPipelineBuilder& SetDepth(
        vk::Bool32 depthTest = vk::False, vk::Bool32 depthWrite = vk::False,
        vk::CompareOp compare = vk::CompareOp::eLessOrEqual);

    // TODO: specify stencil - pipeline settings
    GraphicsPipelineBuilder& SetStencil(vk::Bool32 stencil = vk::False);

    void Clear();

    vk::Pipeline Build(vk::Device device);
};