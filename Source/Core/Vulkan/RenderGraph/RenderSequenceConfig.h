#pragma once

#include <optional>

#include "ArgumentTypes.h"
#include "Core/Vulkan/Native/Buffer.h"
#include "RenderPassBindingInfo.h"
#include "RenderSequence.h"

namespace IntelliDesign_NS::Vulkan::Core {

class IPassConfig;
class RenderPassConfig;
class CopyPassConfig;
class ExecutorConfig;

class RenderSequenceConfig {
public:
    ~RenderSequenceConfig();
    RenderPassConfig& AddRenderPass(const char* passName,
                                    const char* pipelineName);

    RenderPassConfig& AddRenderPass(const char* passName,
                                    PipelineLayout const* pipelineLayout);

    CopyPassConfig& AddCopyPass(const char* passName);

    ExecutorConfig& AddExecutor(const char* passName);

    void Compile(RenderSequence& result);

private:
    Type_STLVector<UniquePtr<IPassConfig>> mPassConfigs;
};

class IPassConfig {
public:
    explicit IPassConfig(const char* passName);

    virtual ~IPassConfig() = default;

protected:
    friend RenderSequenceConfig;

    virtual void Compile(RenderSequence& result) = 0;

    Type_STLString mPassName;
};

class RenderPassConfig : public IPassConfig {
    using Self = RenderPassConfig;

public:
    RenderPassConfig(const char* passName, const char* pipelineName);

    RenderPassConfig(const char* passName,
                     PipelineLayout const* pipelineLayout);

    virtual ~RenderPassConfig() override = default;

    Self& SetBinding(const char* param, const char* argument);

    Self& SetBinding(const char* param,
                     RenderPassBinding::BindlessDescBufInfo bindless);

    Self& SetRenderArea(vk::Rect2D const& area);
    Self& SetViewport(vk::Viewport const& viewport);
    Self& SetScissor(vk::Rect2D const& scissor);
    Self& SetDGCPipelineInfo(DGCPipelineInfo const& info);
    Self& SetDGCSeqBufs(Type_STLVector<const char*> const& buffers);
    Self& SetRTVClearValues(
        Type_STLVector<::std::optional<vk::ClearColorValue>> const& values);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence& result) override;

    PipelineLayout const* mPipelineLayout {nullptr};

    Type_STLString mPipelineName;

    Type_STLVector<::std::pair<Type_STLString, Type_STLString>> mConfigs;
    ::std::optional<
        ::std::pair<Type_STLString, RenderPassBinding::BindlessDescBufInfo>>
        mBindlessDesc;

    Type_STLVector<Type_STLString> mDGCSeqBufs {};

    ::std::optional<vk::Rect2D> mRenderArea;
    ::std::optional<vk::Viewport> mViewport;
    ::std::optional<vk::Rect2D> mScissor;
    ::std::optional<DGCPipelineInfo> mDgcPipelineInfo;
    Type_STLVector<::std::optional<vk::ClearColorValue>> mRTVClearValues;
};

class CopyPassConfig : public IPassConfig {
    using Self = CopyPassConfig;
    using Type_Region =
        ::std::variant<vk::BufferCopy2, vk::BufferImageCopy2, vk::ImageCopy2>;

public:
    struct CopyInfo {
        const char* src;
        const char* dst;
        Type_Region region;
        bool isAsync {false};
    };

public:
    CopyPassConfig(const char* passName);

    virtual ~CopyPassConfig() override = default;

    Self& SetBinding(CopyInfo const& info);
    Self& SetBinding(Type_STLVector<CopyInfo> const& info);
    Self& SetClearBuffer(Type_STLVector<const char*> const& buffers);

    Self& SetAsync(bool isAsync);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence& result) override;

    Type_STLVector<const char*> mBuffersToClear {};
    Type_STLVector<CopyInfo> mConfigs;

    bool mIsAync {false};
};

class ExecutorConfig : public IPassConfig {
    using ResourceStateInfo = RenderPassBindingInfo_Executor::ResourceStateInfo;

public:
    using ResourceStateInfos =
        RenderPassBindingInfo_Executor::ResourceStateInfos;

private:
    using Self = ExecutorConfig;

    // TODO: add frame index param
    using Type_Func = RenderPassBindingInfo_Executor::Type_Func;

public:
    ExecutorConfig(const char* passName);

    virtual ~ExecutorConfig() override = default;

    Self& SetBinding(ResourceStateInfos const& binding);

    void SetExecution(Type_Func&& func);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence& result) override;

    Type_STLVector<ResourceStateInfos> mResourceStateInfos;
    Type_Func mExecution;
};

}  // namespace IntelliDesign_NS::Vulkan::Core