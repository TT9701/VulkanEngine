#include "RenderSequenceConfig.h"

#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderSequenceConfig::~RenderSequenceConfig() {
    mPassConfigs.clear();
}

RenderPassConfig& RenderSequenceConfig::AddRenderPass(
    const char* passName, const char* pipelineName) {
    mPassConfigs.emplace_back(
        MakeUnique<RenderPassConfig>(passName, pipelineName));
    return *dynamic_cast<RenderPassConfig*>(mPassConfigs.back().get());
}

CopyPassConfig& RenderSequenceConfig::AddCopyPass(const char* passName) {
    mPassConfigs.emplace_back(MakeUnique<CopyPassConfig>(passName));
    return *dynamic_cast<CopyPassConfig*>(mPassConfigs.back().get());
}

ExecutorConfig& RenderSequenceConfig::AddExecutor(const char* passName) {
    mPassConfigs.emplace_back(MakeUnique<ExecutorConfig>(passName));
    return *dynamic_cast<ExecutorConfig*>(mPassConfigs.back().get());
}

void RenderSequenceConfig::Compile(RenderSequence& result) {
    for (auto& passConfig : mPassConfigs) {
        passConfig->Compile(result);
    }

    result.GenerateBarriers();
}

RenderPassConfig::RenderPassConfig(const char* passName,
                                   const char* pipelineName)
    : IPassConfig {passName}, mPipelineName(pipelineName) {}

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

RenderPassConfig::Self& RenderPassConfig::SetBinding(Buffer* argumentBuffer) {
    mArgumentBuffer = argumentBuffer;
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

void RenderPassConfig::SetExecuteInfo(ExecuteType type,
                                      std::optional<uint32_t> startIdx,
                                      std::optional<uint32_t> count) {
    mExecuteType = type;
    mStartIdx = startIdx ? *startIdx : 0;
    mDrawCount = count ? *count
               : mArgumentBuffer
                   ? mArgumentBuffer->GetCount() - mStartIdx
                   : throw ::std::runtime_error("No argument buffer provided");
}

void RenderPassConfig::Compile(RenderSequence& result) {
    auto type =
        result.mPipelineMgr.GetPipeline(mPipelineName.c_str()).GetType();

    switch (type) {
        case PipelineType::Graphics:
            result.AddRenderPass(mPassName.c_str(), RenderQueueType::Graphics);
            break;
        case PipelineType::Compute:
            result.AddRenderPass(mPassName.c_str(), RenderQueueType::Compute);
            break;
    }

    auto& pso = *dynamic_cast<RenderPassBindingInfo_PSO*>(
        result.FindPass(mPassName.c_str()).binding.get());

    pso.SetPipeline(mPipelineName.c_str());

    if (!mPushConstants.empty()) {
        for (auto const& pc : mPushConstants) {
            pso[pc.first.c_str()] = pc.second;
        }
    }

    if (mBindlessDesc) {
        pso[mBindlessDesc->first.c_str()] = mBindlessDesc->second;
    }

    for (auto const& [param, argument] : mConfigs) {
        if (param == "_Depth_") {
            pso[RenderPassBinding::Type::DSV] = argument;
            continue;
        }
        pso[param.c_str()] = argument;
    }

    if (mRenderArea) {
        pso[RenderPassBinding::Type::RenderInfo] =
            RenderPassBinding::RenderInfo {*mRenderArea, 1, 0};
    }

    pso.GenerateMetaData();

    auto& dcMgr = pso.GetDrawCallManager();

    if (mViewport) {
        dcMgr.AddArgument_Viewport(0, {*mViewport});
    }

    if (mScissor) {
        dcMgr.AddArgument_Scissor(0, {*mScissor});
    }

    switch (mExecuteType) {
        case ExecuteType::Dispatch: {
            dcMgr.AddArgument_DispatchIndirect(
                mArgumentBuffer->GetHandle(),
                mStartIdx * mArgumentBuffer->GetStride());
            break;
            case ExecuteType::Draw: {
                dcMgr.AddArgument_DrawIndirect(
                    mArgumentBuffer->GetHandle(),
                    mStartIdx * mArgumentBuffer->GetStride(), mDrawCount,
                    mArgumentBuffer->GetStride());
                break;
            }
            case ExecuteType::DrawIndexed: {
                dcMgr.AddArgument_DrawIndexedIndirect(
                    mArgumentBuffer->GetHandle(),
                    mStartIdx * mArgumentBuffer->GetStride(), mDrawCount,
                    mArgumentBuffer->GetStride());
                break;
            }
            case ExecuteType::DrawMeshTask: {
                dcMgr.AddArgument_DrawMeshTasksIndirect(
                    mArgumentBuffer->GetHandle(),
                    mStartIdx * mArgumentBuffer->GetStride(), mDrawCount,
                    mArgumentBuffer->GetStride());
                break;
            }
        }
    }
}

CopyPassConfig::CopyPassConfig(const char* passName) : IPassConfig(passName) {}

CopyPassConfig::Self& CopyPassConfig::SetBinding(CopyInfo const& info) {
    mConfigs.push_back(info);
    return *this;
}

CopyPassConfig::Self& CopyPassConfig::SetAsync(bool isAsync) {
    mIsAync = isAsync;
    return *this;
}

void CopyPassConfig::Compile(RenderSequence& result) {
    auto& copyPass = result.AddCopyPass(mPassName.c_str());

    auto getResType = [&result](const char* name) {
        return result.mResMgr[name].GetType();
    };

    for (auto const& config : mConfigs) {
        auto srcType = getResType(config.src);
        auto dstType = getResType(config.dst);

        if (srcType == RenderResource::Type::Buffer) {
            if (dstType == RenderResource::Type::Buffer) {
                auto region = ::std::get<vk::BufferCopy2>(config.region);
                copyPass.CopyBufferToBuffer(config.src, config.dst, region);
            } else {
                auto region = ::std::get<vk::BufferImageCopy2>(config.region);
                copyPass.CopyBufferToImage(config.src, config.dst, region);
            }
        } else {
            if (dstType == RenderResource::Type::Buffer) {
                auto region = ::std::get<vk::BufferImageCopy2>(config.region);
                copyPass.CopyImageToBuffer(config.src, config.dst, region);
            } else {
                auto region = ::std::get<vk::ImageCopy2>(config.region);
                copyPass.CopyImageToImage(config.src, config.dst, region);
            }
        }
    }

    copyPass.GenerateMetaData();
}

ExecutorConfig::ExecutorConfig(const char* passName) : IPassConfig(passName) {}

ExecutorConfig::Self& ExecutorConfig::SetBinding(
    ResourceStateInfos const& binding) {
    mResourceStateInfos.push_back(binding);
    return *this;
}

void ExecutorConfig::SetExecution(Type_Func&& func) {
    mExecution = ::std::move(func);
}

void ExecutorConfig::Compile(RenderSequence& result) {
    auto& executor = result.AddExecutor(mPassName.c_str());
    executor.AddExecution(mResourceStateInfos, ::std::move(mExecution));
}

}  // namespace IntelliDesign_NS::Vulkan::Core