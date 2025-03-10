#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Shader.h"

namespace IntelliDesign_NS::Vulkan::Core {

enum class PipelineType { Graphics, Compute };

class Context;

class PipelineLayout {
public:
    PipelineLayout(VulkanContext& context, ShaderProgram& program,
                   vk::PipelineLayoutCreateFlags flags = {},
                   void* pNext = nullptr);
    ~PipelineLayout();
    CLASS_MOVABLE_ONLY(PipelineLayout);

public:
    vk::PipelineLayout GetHandle() const { return mLayout; }

    ShaderProgram& GetShaderProgram() const;

    Type_STLVector<DescriptorSetLayout*> GetDescSetLayoutDatas() const;

    Type_STLVector<vk::PushConstantRange> GetPCRanges() const;

    ShaderProgram::Type_CombinedPushContant const& GetCombinedPushContant()
        const;

    Type_STLVector<Type_STLString> const& GetRTVNames() const;

private:
    vk::PipelineLayout CreateLayout(vk::PipelineLayoutCreateFlags flags,
                                    void* pNext) const;

private:
    VulkanContext& mContext;
    ShaderProgram& mProgram;

    vk::PipelineLayout mLayout;
};

class PipelineManager;

class Pipeline {
public:
    Pipeline(VulkanContext& context, vk::GraphicsPipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    Pipeline(VulkanContext& context, vk::ComputePipelineCreateInfo const& info,
             vk::PipelineCache cache = {});
    ~Pipeline();

    CLASS_NO_COPY(Pipeline);

public:
    vk::Pipeline GetHandle() const;
    PipelineType GetType() const;

    friend PipelineManager;

private:
    VulkanContext& mContext;
    PipelineType mType;
    vk::Pipeline mHandle;
};

template <PipelineType Type>
class PipelineBuilder;

template <>
class PipelineBuilder<PipelineType::Graphics> {
public:
    PipelineBuilder(PipelineManager& manager);
    ~PipelineBuilder() = default;
    CLASS_MOVABLE_ONLY(PipelineBuilder);

public:
    using Self = PipelineBuilder<PipelineType::Graphics>;
    Self& SetLayout(vk::PipelineLayout layout);
    Self& SetShaderProgram(ShaderProgram* program);
    Self& SetInputTopology(vk::PrimitiveTopology topology);
    Self& SetPolygonMode(vk::PolygonMode mode);
    Self& SetCullMode(vk::CullModeFlags cullMode, vk::FrontFace frontFace);

    // TODO: finish multisampling - pipeline settings
    Self& SetMultisampling(
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    // TODO: finish blending - pipeline settings
    Self& SetBlending(vk::Bool32 enable = vk::False);

    Self& SetColorAttachmentFormat(vk::Format format);
    Self& SetDepthStencilFormat(vk::Format format);
    Self& SetDepth(vk::Bool32 depthTest = vk::False,
                   vk::Bool32 depthWrite = vk::False,
                   vk::CompareOp compare = vk::CompareOp::eLessOrEqual);

    // TODO: specify stencil - pipeline settings
    Self& SetStencil(vk::Bool32 stencil = vk::False);
    Self& SetBaseHandle(vk::Pipeline baseHandle);
    Self& SetBaseIndex(int32_t index);
    Self& SetFlags(vk::PipelineCreateFlags flags);

    SharedPtr<Pipeline> Build(const char* name, vk::PipelineCache cache = {},
                              void* pNext = nullptr);

    void Clear();

private:
    PipelineManager& mManager;
    ShaderProgram* pProgram;

    Type_STLVector<vk::PipelineShaderStageCreateInfo> mShaderStages {};
    Type_STLVector<vk::DescriptorSetLayout> mDescriptorSetLayouts {};

    vk::PipelineInputAssemblyStateCreateInfo mInputAssembly;
    vk::PipelineRasterizationStateCreateInfo mRasterizer;
    vk::PipelineColorBlendAttachmentState mColorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo mMultisampling;
    vk::PipelineLayout mPipelineLayout;
    vk::PipelineDepthStencilStateCreateInfo mDepthStencil;
    vk::PipelineRenderingCreateInfo mRenderInfo;
    vk::Format mColorAttachmentformat;
    vk::Pipeline mBaseHandle {};
    int32_t mBaseIndex {};
    vk::PipelineCreateFlags mFlags {};
};

template <>
class PipelineBuilder<PipelineType::Compute> {
public:
    PipelineBuilder(PipelineManager& manager);
    ~PipelineBuilder() = default;
    CLASS_MOVABLE_ONLY(PipelineBuilder);

public:
    using Self = PipelineBuilder<PipelineType::Compute>;

    Self& SetShaderProgram(ShaderProgram* program);
    Self& SetLayout(vk::PipelineLayout pipelineLayout);
    Self& SetFlags(vk::PipelineCreateFlags flags);
    Self& SetBaseHandle(vk::Pipeline baseHandle);
    Self& SetBaseIndex(int32_t index);

    SharedPtr<Pipeline> Build(const char* name, vk::PipelineCache cache = {},
                              void* pNext = nullptr);

    void Clear();

private:
    PipelineManager& mManager;
    ShaderProgram* pProgram;

    vk::PipelineShaderStageCreateInfo mStageInfo {};
    vk::PipelineLayout mPipelineLayout {};
    vk::PipelineCreateFlags mFlags {};
    vk::Pipeline mBaseHandle {};
    int32_t mBaseIndex {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core