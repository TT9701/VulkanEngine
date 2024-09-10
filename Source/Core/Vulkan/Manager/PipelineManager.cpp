#include "PipelineManager.hpp"

#include <spirv_glsl.hpp>

#include "Context.hpp"
#include "Core/Vulkan/Native/Shader.hpp"
#include "DescriptorManager.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineBuilder<PipelineType::Graphics>::PipelineBuilder(
    PipelineManager* manager, DescriptorManager* descriptorManager)
    : pManager(manager), pDescriptorManager(descriptorManager) {
    Clear();
}

PipelineBuilder<PipelineType::Graphics>&
PipelineBuilder<PipelineType::Graphics>::SetLayout(vk::PipelineLayout layout) {
    mPipelineLayout = layout;
    return *this;
}

PipelineBuilder<PipelineType::Graphics>& PipelineBuilder<
    PipelineType::Graphics>::SetShaders(::std::span<Shader*> shaders) {
    mShaderStages.clear();
    for (const auto& shader : shaders) {
        shader->GetMutex().lock();
        pShaders.push_back(shader);
        mShaderStages.push_back(shader->GetStageInfo());
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

    auto shaderStatus = pManager->ReflectShaderStats(
        pipelineLayoutName.c_str(), pDescriptorManager, pShaders);

    auto pipelineLayout = pManager->CreateLayout(pipelineLayoutName.c_str(),
                                                 shaderStatus.descSetLayouts,
                                                 shaderStatus.pushContant);

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

    for (const auto& modules : pShaders) {
        modules->GetMutex().unlock();
    }
    pManager->pContext->SetName(pipeline->GetHandle(), pipelineName);
    pManager->mGraphicsPipelines.emplace(pipelineName, pipeline);

    Clear();

    return pipeline;
}

void PipelineBuilder<PipelineType::Graphics>::Clear() {
    mShaderStages.clear();
    pShaders.clear();
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
    PipelineManager* manager, DescriptorManager* descriptorManager)
    : pManager(manager), pDescriptorManager(descriptorManager) {
    Clear();
}

PipelineBuilder<PipelineType::Compute>&
PipelineBuilder<PipelineType::Compute>::SetShader(Shader* shader) {
    shader->GetMutex().lock();
    pShader = shader;
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

    Type_STLVector<Shader*> shaders {pShader};
    auto shaderStatus = pManager->ReflectShaderStats(
        pipelineLayoutName.c_str(), pDescriptorManager, shaders);

    auto pipelineLayout = pManager->CreateLayout(pipelineLayoutName.c_str(),
                                                 shaderStatus.descSetLayouts,
                                                 shaderStatus.pushContant);

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

    pShader->GetMutex().unlock();
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
    const char* name, ::std::span<vk::DescriptorSetLayout> setLayouts,
    ::std::span<vk::PushConstantRange> pushContants,
    vk::PipelineLayoutCreateFlags flags, void* pNext) {
    const auto ptr = MakeShared<PipelineLayout>(pContext, setLayouts,
                                                pushContants, flags, pNext);

    pContext->SetName(ptr->GetHandle(), name);
    mPipelineLayouts.emplace(name, ptr);

    return ptr;
}

vk::PipelineLayout PipelineManager::GetLayoutHandle(const char* name) const {
    return mPipelineLayouts.at(ParsePipelineLayoutName(name))->GetHandle();
}

vk::Pipeline PipelineManager::GetComputePipelineHandle(const char* name) const {

    return mComputePipelines.at(ParsePipelineName(name))->GetHandle();
}

vk::Pipeline PipelineManager::GetGraphicsPipelineHandle(
    const char* name) const {
    return mGraphicsPipelines.at(ParsePipelineName(name))->GetHandle();
}

PipelineManager::Type_CPBuilder PipelineManager::GetComputePipelineBuilder(
    DescriptorManager* descManager) {
    return Type_CPBuilder {this, descManager};
}

PipelineManager::Type_GPBuilder PipelineManager::GetGraphicsPipelineBuilder(
    DescriptorManager* descManager) {
    return Type_GPBuilder {this, descManager};
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

struct Comp {
    template <typename T>
    bool operator()(const T& l, const T& r) const {
        if (l.setIdx != r.setIdx) {
            return l.setIdx < r.setIdx;
        }
        return l.bindingIdx < r.bindingIdx;
    }
};

ShaderStats PipelineManager::ReflectShaderStats(const char* pipelineName,
                                                DescriptorManager* descManager,
                                                ::std::span<Shader*> shaders) {

    Type_STLVector<DescriptorSetLayoutData> datas {};

    for (auto const& shader : shaders) {
        datas.insert(
            datas.end(),
            ::std::make_move_iterator(shader->GetDescSetLayoutDatas().begin()),
            ::std::make_move_iterator(shader->GetDescSetLayoutDatas().end()));
    }

    std::ranges::sort(datas, Comp {});

    auto makeUniqueSet =
        [&](const Type_STLVector<DescriptorSetLayoutData>::iterator& prev,
            const Type_STLVector<DescriptorSetLayoutData>::iterator& last) {
            Type_STLVector<DescriptorSetLayoutData> uniqueSet {};
            for (auto it = prev; it != last; ++it) {
                uniqueSet.push_back(*it);
            }
            return uniqueSet;
        };

    Type_STLVector<Type_STLVector<DescriptorSetLayoutData>> uniqueSets {};
    {
        auto prev = datas.begin();
        auto last = ++datas.begin();

        while (prev != datas.end()) {
            if (last == datas.end()) {
                uniqueSets.push_back(makeUniqueSet(prev, last));
                break;
            }
            if (last->setIdx == prev->setIdx) {
                ++last;
                continue;
            }
            uniqueSets.push_back(makeUniqueSet(prev, last));
            prev = last;
        }
    }

    auto mergeBinding =
        [&](Type_STLVector<DescriptorSetLayoutData> const& bindings) {
            DescriptorSetLayoutData data {bindings[0]};
            Type_STLString prefix {};
            for (uint32_t i = 1; i < bindings.size(); ++i) {
                data.stage |= bindings[i].stage;
            }
            prefix += vk::to_string(data.stage) + "_";
            data.name = prefix + data.name;
            return data;
        };

    Type_STLVector<Type_STLVector<DescriptorSetLayoutData>>
        uniqueBindingSets {};
    for (auto& set : uniqueSets) {
        auto prev = set.begin();
        auto last = ++set.begin();
        Type_STLVector<DescriptorSetLayoutData> uniqueBindingSet {};
        while (prev != set.end()) {
            if (last == set.end()) {
                uniqueBindingSet.push_back(
                    mergeBinding(makeUniqueSet(prev, last)));

                break;
            }
            if (last->bindingIdx == prev->bindingIdx) {
                ++last;
                continue;
            }
            uniqueBindingSet.push_back(mergeBinding(makeUniqueSet(prev, last)));

            prev = last;
        }
        uniqueBindingSets.push_back(uniqueBindingSet);
    }

    datas.clear();
    for (auto& set : uniqueBindingSets) {
        for (auto& data : set) {
            datas.emplace_back(data);
        }
    }

    Type_STLVector<vk::DescriptorSetLayout> descSetLayouts {};
    if (!datas.empty()) {
        auto names = descManager->CreateDescLayouts(pipelineName, datas);

        for (auto& name : names) {
            descSetLayouts.emplace_back(
                descManager->GetDescSetLayout((pipelineName + name).c_str())
                    ->GetHandle());
        }
    }

    // push contants
    Type_STLVector<vk::PushConstantRange> pushConstants;
    for (auto const& shader : shaders) {
        auto data = shader->GetPushContantData();
        if (data.has_value()) {
            pushConstants.push_back(data.value());
        }
    }

    return {descSetLayouts, pushConstants};
}

Type_STLString PipelineManager::ParsePipelineName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName} + "_pipeline";
}

Type_STLString PipelineManager::ParsePipelineLayoutName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName} + "_pipeline_layout";
}

}  // namespace IntelliDesign_NS::Vulkan::Core