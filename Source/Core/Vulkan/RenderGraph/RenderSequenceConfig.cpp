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

RenderPassConfig& RenderSequenceConfig::AddRenderPass(
    const char* passName, PipelineLayout const* pipelineLayout) {
    mPassConfigs.emplace_back(
        MakeUnique<RenderPassConfig>(passName, pipelineLayout));
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

IPassConfig::IPassConfig(const char* passName) : mPassName(passName) {}

RenderPassConfig::RenderPassConfig(const char* passName,
                                   const char* pipelineName)
    : IPassConfig {passName}, mPipelineName(pipelineName) {}

RenderPassConfig::RenderPassConfig(const char* passName,
                                   PipelineLayout const* pipelineLayout)
    : IPassConfig(passName), mPipelineLayout(pipelineLayout) {}

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

RenderPassConfig::Self& RenderPassConfig::SetDGCPipelineInfo(
    DGCPipelineInfo const& info) {
    mDgcPipelineInfo = info;
    return *this;
}

RenderPassConfig::Self& RenderPassConfig::SetDGCSeqBufs(
    Type_STLVector<const char*> const& buffers) {
    mDGCSeqBufs.reserve(buffers.size());
    for (auto const& buf : buffers) {
        mDGCSeqBufs.emplace_back(buf);
    }
    return *this;
}

void RenderPassConfig::Compile(RenderSequence& result) {
    if (mPipelineLayout) {
        result.AddRenderPass(mPassName.c_str(), mPipelineLayout)
            .GenerateLayoutData();
    } else {
        result.AddRenderPass(mPassName.c_str())
            .SetPipeline(mPipelineName.c_str());
    }

    auto& pso = *dynamic_cast<RenderPassBindingInfo_PSO*>(
        result.FindPass(mPassName.c_str()).binding.get());

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

    auto& dcMgr = pso.GetDrawCallManager();

    if (mDgcPipelineInfo) {
        dcMgr.AddArgument_DGCPipelineInfo(*mDgcPipelineInfo);
    } else {
        if (mViewport) {
            dcMgr.AddArgument_Viewport(0, {*mViewport});
        }

        if (mScissor) {
            dcMgr.AddArgument_Scissor(0, {*mScissor});
        }
    }

    if (!mDGCSeqBufs.empty()) {
        Type_STLVector<Type_STLString> tmp {};
        tmp.reserve(mDGCSeqBufs.size());
        for (auto const& buf : mDGCSeqBufs) {
            tmp.emplace_back(buf);
        }
        pso[RenderPassBinding::Type::DGCSeqBuf] = tmp;
    }

    pso.GenerateMetaData();
}

CopyPassConfig::CopyPassConfig(const char* passName) : IPassConfig(passName) {}

CopyPassConfig::Self& CopyPassConfig::SetBinding(CopyInfo const& info) {
    mConfigs.push_back(info);
    return *this;
}

CopyPassConfig::Self& CopyPassConfig::SetBinding(
    Type_STLVector<CopyInfo> const& info) {
    mConfigs.insert(mConfigs.end(), info.begin(), info.end());
    return *this;
}

CopyPassConfig::Self& CopyPassConfig::SetClearBuffer(
    Type_STLVector<const char*> const& buffers) {
    mBuffersToClear = buffers;
    return *this;
}

CopyPassConfig::Self& CopyPassConfig::SetAsync(bool isAsync) {
    mIsAync = isAsync;
    return *this;
}

void CopyPassConfig::Compile(RenderSequence& result) {
    auto& copyPass = result.AddCopyPass(mPassName.c_str());

    if (!mBuffersToClear.empty())
        copyPass.ClearBuffers(mBuffersToClear);

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