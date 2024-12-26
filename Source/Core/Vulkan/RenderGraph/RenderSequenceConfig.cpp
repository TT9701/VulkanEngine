#include "RenderSequenceConfig.h"

#include "Core/Vulkan/Manager/PipelineManager.h"
#include "RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderPassConfig& RenderSequenceConfig::AddRenderPass(
    const char* passName, const char* pipelineName) {
    mPassConfigs.emplace_back(passName, pipelineName);
    return mPassConfigs.back();
}

void RenderSequenceConfig::Compile(RenderSequence& result) {
    for (auto& passConfig : mPassConfigs) {
        auto type =
            result.mPipelineMgr.GetPipeline(passConfig.mPipelineName.c_str())
                .GetType();
        switch (type) {
            case PipelineType::Graphics: {
                auto& pass = result.AddRenderPass(passConfig.mPassName.c_str(),
                                                  RenderQueueType::Graphics);
                passConfig.Compile(pass);
                break;
            }
            case PipelineType::Compute: {
                auto& pass = result.AddRenderPass(passConfig.mPassName.c_str(),
                                                  RenderQueueType::Compute);
                passConfig.Compile(pass);
                break;
            }
        }
    }
}

RenderPassConfig::RenderPassConfig(const char* passName,
                                   const char* pipelineName)
    : mPassName(passName), mPipelineName(pipelineName) {}

RenderPassConfig::Self& RenderPassConfig::SetBinding(const char* param,
                                                     const char* argument) {
    mConfigs.emplace_back(param, argument);
    return *this;
}

RenderPassConfig::Self& RenderPassConfig::SetBinding(
    const char* param, RenderPassBinding::BindlessDescBufInfo bindless) {
    mBindlessDesc = {param, bindless};
    return *this;
}

RenderPassConfig::Self& RenderPassConfig::SetRenderArea(
    vk::Rect2D const& area) {
    mRenderArea = area;
    return *this;
}

RenderPassConfig::Self& RenderPassConfig::SetViewport(
    vk::Viewport const& viewport) {
    mViewport = viewport;
    return *this;
}

RenderPassConfig::Self& RenderPassConfig::SetScissor(
    vk::Rect2D const& scissor) {
    mScissor = scissor;
    return *this;
}

void RenderPassConfig::Compile(RenderSequence::RenderPassBindingInfo& info) {
    auto& pso = *info.pso;

    pso.SetPipeline(mPipelineName.c_str());

    if (mPushConstants) {
        pso["constants"] = mPushConstants.value();
    }

    if (mBindlessDesc) {
        pso[mBindlessDesc->first.c_str()] = mBindlessDesc->second;
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core