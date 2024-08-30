#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

enum class PipelineType { Graphics, Compute };

class Context;
class Shader;

class PipelineLayout {
public:
    PipelineLayout(Context* context,
                   ::std::span<vk::DescriptorSetLayout> setLayouts,
                   ::std::span<vk::PushConstantRange> pushContants,
                   vk::PipelineLayoutCreateFlags flags = {},
                   void* pNext = nullptr);
    ~PipelineLayout();
    MOVABLE_ONLY(PipelineLayout);

public:
    vk::PipelineLayout GetHandle() const { return mLayout; }

private:
    vk::PipelineLayout CreateLayout(
        ::std::span<vk::DescriptorSetLayout> setLayouts,
        ::std::span<vk::PushConstantRange> pushContants,
        vk::PipelineLayoutCreateFlags flags, void* pNext) const;

private:
    Context* pContext;

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

class PipelineManager;

template <PipelineType Type>
class PipelineBuilder;

template <>
class PipelineBuilder<PipelineType::Graphics> {
public:
    PipelineBuilder(PipelineManager* manager);
    ~PipelineBuilder() = default;
    MOVABLE_ONLY(PipelineBuilder);

public:
    PipelineBuilder& SetLayout(vk::PipelineLayout layout);

    PipelineBuilder& SetShaders(::std::span<Shader> shaders);

    PipelineBuilder& SetInputTopology(vk::PrimitiveTopology topology);

    PipelineBuilder& SetPolygonMode(vk::PolygonMode mode);

    PipelineBuilder& SetCullMode(vk::CullModeFlags cullMode,
                                 vk::FrontFace frontFace);

    // TODO: finish multisampling - pipeline settings
    PipelineBuilder& SetMultisampling(
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    // TODO: finish blending - pipeline settings
    PipelineBuilder& SetBlending(vk::Bool32 enable = vk::False);

    PipelineBuilder& SetColorAttachmentFormat(vk::Format format);

    PipelineBuilder& SetDepthStencilFormat(vk::Format format);

    PipelineBuilder& SetDepth(
        vk::Bool32 depthTest = vk::False, vk::Bool32 depthWrite = vk::False,
        vk::CompareOp compare = vk::CompareOp::eLessOrEqual);

    // TODO: specify stencil - pipeline settings
    PipelineBuilder& SetStencil(vk::Bool32 stencil = vk::False);

    SharedPtr<Pipeline<PipelineType::Graphics>> Build(
        ::std::string_view name, vk::PipelineCache cache = {},
        void* pNext = nullptr);

    void Clear();

private:
    PipelineManager* pManager;

    Type_STLVector<vk::PipelineShaderStageCreateInfo> mShaderStages {};

    vk::PipelineInputAssemblyStateCreateInfo mInputAssembly;
    vk::PipelineRasterizationStateCreateInfo mRasterizer;
    vk::PipelineColorBlendAttachmentState mColorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo mMultisampling;
    vk::PipelineLayout mPipelineLayout;
    vk::PipelineDepthStencilStateCreateInfo mDepthStencil;
    vk::PipelineRenderingCreateInfo mRenderInfo;
    vk::Format mColorAttachmentformat;
};

template <>
class PipelineBuilder<PipelineType::Compute> {
public:
    PipelineBuilder(PipelineManager* manager);
    ~PipelineBuilder() = default;
    MOVABLE_ONLY(PipelineBuilder);

public:
    PipelineBuilder& SetShader(Shader const& shader);

    PipelineBuilder& SetLayout(vk::PipelineLayout pipelineLayout);

    PipelineBuilder& SetFlags(vk::PipelineCreateFlags flags);

    PipelineBuilder& SetBaseHandle(vk::Pipeline baseHandle);

    PipelineBuilder& SetBaseIndex(int32_t index);

    SharedPtr<Pipeline<PipelineType::Compute>> Build(
        ::std::string_view name, vk::PipelineCache cache = {},
        void* pNext = nullptr);

    void Clear();

private:
    PipelineManager* pManager;

    vk::PipelineShaderStageCreateInfo mStageInfo {};
    vk::PipelineLayout mPipelineLayout {};
    vk::PipelineCreateFlags mFlags {};
    vk::Pipeline mBaseHandle {};
    int32_t mBaseIndex {};
};

/**
 * Use CreateLayout() to build Pipeline Layouts
 * Use PipelineBuilder to build Pipelines
 */
class PipelineManager {
    using Type_CPBuilder = PipelineBuilder<PipelineType::Compute>;
    using Type_GPBuilder = PipelineBuilder<PipelineType::Graphics>;
    using Type_ComPipeline = Pipeline<PipelineType::Compute>;
    using Type_GrapPipeline = Pipeline<PipelineType::Graphics>;

    using Type_PipelineLayouts =
        Type_STLUnorderedMap_String<SharedPtr<PipelineLayout>>;
    using Type_ComputePipelines =
        Type_STLUnorderedMap_String<SharedPtr<Type_ComPipeline>>;
    using Type_GraphicsPipelines =
        Type_STLUnorderedMap_String<SharedPtr<Type_GrapPipeline>>;

public:
    PipelineManager(Context* contex);
    ~PipelineManager() = default;
    MOVABLE_ONLY(PipelineManager);

    friend Type_CPBuilder;
    friend Type_GPBuilder;

public:
    SharedPtr<PipelineLayout> CreateLayout(
        ::std::string_view name,
        ::std::span<vk::DescriptorSetLayout> setLayouts,
        ::std::span<vk::PushConstantRange> pushContants = {},
        vk::PipelineLayoutCreateFlags flags = {}, void* pNext = nullptr);

public:
    vk::PipelineLayout GetLayoutHandle(::std::string_view name) const;

    vk::Pipeline GetComputePipeline(::std::string_view name) const;
    vk::Pipeline GetGraphicsPipeline(::std::string_view name) const;

    Type_CPBuilder& GetComputePipelineBuilder();
    Type_GPBuilder& GetGraphicsPipelineBuilder();

private:
    Context* pContext;

    Type_CPBuilder mComputePipelineBuilder;
    Type_GPBuilder mGraphicsPipelineBuilder;

    Type_PipelineLayouts mPipelineLayouts;
    Type_ComputePipelines mComputePipelines;
    Type_GraphicsPipelines mGraphicsPipelines;
};

}  // namespace IntelliDesign_NS::Vulkan::Core