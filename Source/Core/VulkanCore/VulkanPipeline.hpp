#pragma once

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

enum class PipelineType { Graphics, Compute };

class VulkanContext;
class VulkanShader;

class VulkanPipelineLayout {
public:
    VulkanPipelineLayout(
        VulkanContext*                                 context,
        vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
        vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
        vk::PipelineLayoutCreateFlags flags = {}, void* pNext = nullptr);
    ~VulkanPipelineLayout();
    MOVABLE_ONLY(VulkanPipelineLayout);

public:
    vk::PipelineLayout GetHandle() const { return mLayout; }

private:
    vk::PipelineLayout CreateLayout(
        vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
        vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
        vk::PipelineLayoutCreateFlags flags, void* pNext) const;

private:
    VulkanContext* pContext;

    vk::PipelineLayout mLayout;
};

template <PipelineType Type>
class VulkanPipeline;

template <>
class VulkanPipeline<PipelineType::Graphics> {
public:
    VulkanPipeline(VulkanContext*                        context,
                   vk::GraphicsPipelineCreateInfo const& info,
                   vk::PipelineCache                     cache = {});
    ~VulkanPipeline();
    MOVABLE_ONLY(VulkanPipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache                     cache,
        vk::GraphicsPipelineCreateInfo const& info) const;

private:
    VulkanContext* pContext;

    vk::Pipeline mPipeline;
};

template <>
class VulkanPipeline<PipelineType::Compute> {
public:
    VulkanPipeline(VulkanContext*                       context,
                   vk::ComputePipelineCreateInfo const& info,
                   vk::PipelineCache                    cache = {});
    ~VulkanPipeline();
    MOVABLE_ONLY(VulkanPipeline);

public:
    vk::Pipeline GetHandle() const { return mPipeline; }

private:
    vk::Pipeline CreatePipeline(
        vk::PipelineCache                    cache,
        vk::ComputePipelineCreateInfo const& info) const;

private:
    VulkanContext* pContext;

    vk::Pipeline mPipeline;
};

class VulkanPipelineManager;

template <PipelineType Type>
class VulkanPipelineBuilder;

template <>
class VulkanPipelineBuilder<PipelineType::Graphics> {
public:
    VulkanPipelineBuilder(VulkanPipelineManager* manager);
    ~VulkanPipelineBuilder() = default;
    MOVABLE_ONLY(VulkanPipelineBuilder);

public:
    VulkanPipelineBuilder& SetLayout(vk::PipelineLayout layout);

    VulkanPipelineBuilder& SetShaders(::std::span<VulkanShader> shaders);

    VulkanPipelineBuilder& SetInputTopology(vk::PrimitiveTopology topology);

    VulkanPipelineBuilder& SetPolygonMode(vk::PolygonMode mode);

    VulkanPipelineBuilder& SetCullMode(vk::CullModeFlags cullMode,
                                       vk::FrontFace     frontFace);

    // TODO: finish multisampling - pipeline settings
    VulkanPipelineBuilder& SetMultisampling(
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    // TODO: finish blending - pipeline settings
    VulkanPipelineBuilder& SetBlending(vk::Bool32 enable = vk::False);

    VulkanPipelineBuilder& SetColorAttachmentFormat(vk::Format format);

    VulkanPipelineBuilder& SetDepthStencilFormat(vk::Format format);

    VulkanPipelineBuilder& SetDepth(
        vk::Bool32 depthTest = vk::False, vk::Bool32 depthWrite = vk::False,
        vk::CompareOp compare = vk::CompareOp::eLessOrEqual);

    // TODO: specify stencil - pipeline settings
    VulkanPipelineBuilder& SetStencil(vk::Bool32 stencil = vk::False);

    SharedPtr<VulkanPipeline<PipelineType::Graphics>> Build(
        ::std::string const& name, vk::PipelineCache cache = {},
        void* pNext = nullptr);

    void Clear();

private:
    VulkanPipelineManager* pManager;

    ::std::vector<vk::PipelineShaderStageCreateInfo> mShaderStages {};

    vk::PipelineInputAssemblyStateCreateInfo mInputAssembly;
    vk::PipelineRasterizationStateCreateInfo mRasterizer;
    vk::PipelineColorBlendAttachmentState    mColorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo   mMultisampling;
    vk::PipelineLayout                       mPipelineLayout;
    vk::PipelineDepthStencilStateCreateInfo  mDepthStencil;
    vk::PipelineRenderingCreateInfo          mRenderInfo;
    vk::Format                               mColorAttachmentformat;
};

template <>
class VulkanPipelineBuilder<PipelineType::Compute> {
public:
    VulkanPipelineBuilder(VulkanPipelineManager* manager);
    ~VulkanPipelineBuilder() = default;
    MOVABLE_ONLY(VulkanPipelineBuilder);

public:
    VulkanPipelineBuilder& SetShader(VulkanShader const& shader);

    VulkanPipelineBuilder& SetLayout(vk::PipelineLayout pipelineLayout);

    VulkanPipelineBuilder& SetFlags(vk::PipelineCreateFlags flags);

    VulkanPipelineBuilder& SetBaseHandle(vk::Pipeline baseHandle);

    VulkanPipelineBuilder& SetBaseIndex(int32_t index);

    SharedPtr<VulkanPipeline<PipelineType::Compute>> Build(
        ::std::string const& name, vk::PipelineCache cache = {},
        void* pNext = nullptr);

    void Clear();

private:
    VulkanPipelineManager* pManager;

    vk::PipelineShaderStageCreateInfo mStageInfo {};
    vk::PipelineLayout                mPipelineLayout {};
    vk::PipelineCreateFlags           mFlags {};
    vk::Pipeline                      mBaseHandle {};
    int32_t                           mBaseIndex {};
};

/**
 * Use CreateLayout() to build Pipeline Layouts
 * Use PipelineBuilder to build Pipelines
 */
class VulkanPipelineManager {
    using CPBuilder    = VulkanPipelineBuilder<PipelineType::Compute>;
    using GPBuilder    = VulkanPipelineBuilder<PipelineType::Graphics>;
    using ComPipeline  = VulkanPipeline<PipelineType::Compute>;
    using GrapPipeline = VulkanPipeline<PipelineType::Graphics>;

public:
    VulkanPipelineManager(VulkanContext* contex);
    ~VulkanPipelineManager() = default;
    MOVABLE_ONLY(VulkanPipelineManager);

    friend CPBuilder;
    friend GPBuilder;

public:
    SharedPtr<VulkanPipelineLayout> CreateLayout(
        ::std::string const&                           name,
        vk::ArrayProxy<vk::DescriptorSetLayout> const& setLayouts,
        vk::ArrayProxy<vk::PushConstantRange> const&   pushContants,
        vk::PipelineLayoutCreateFlags flags = {}, void* pNext = nullptr);

public:
    vk::PipelineLayout GetLayoutHandle(::std::string const& name) const;

    vk::Pipeline GetComputePipeline(::std::string const& name) const;
    vk::Pipeline GetGraphicsPipeline(::std::string const& name) const;

    CPBuilder& GetComputePipelineBuilder();
    GPBuilder& GetGraphicsPipelineBuilder();

private:
    VulkanContext* pContext;

    CPBuilder mComputePipelineBuilder;
    GPBuilder mGraphicsPipelineBuilder;

    ::std::unordered_map<::std::string, SharedPtr<VulkanPipelineLayout>>
        mPipelineLayouts;

    ::std::unordered_map<::std::string, SharedPtr<ComPipeline>>
        mComputePipelines;

    ::std::unordered_map<::std::string, SharedPtr<GrapPipeline>>
        mGraphicsPipelines;
};