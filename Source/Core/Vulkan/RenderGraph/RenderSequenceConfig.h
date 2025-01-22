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

    CopyPassConfig& AddCopyPass(const char* passName);

    ExecutorConfig& AddExecutor(const char* passName);

    void Compile(RenderSequence& result);

private:
    Type_STLVector<UniquePtr<IPassConfig>> mPassConfigs;
};

class IPassConfig {
public:
    IPassConfig(const char* passName) : mPassName(passName) {}

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

    virtual ~RenderPassConfig() override = default;

    Self& SetBinding(const char* param, const char* argument);

    template <class T>
    Self& SetBinding(const char* name, T* pPushConstants);

    Self& SetBinding(const char* param,
                     RenderPassBinding::BindlessDescBufInfo bindless);

    Self& SetBinding(Buffer* argumentBuffer);

    Self& SetRenderArea(vk::Rect2D const& area);
    Self& SetViewport(vk::Viewport const& viewport);
    Self& SetScissor(vk::Rect2D const& scissor);

    enum class ExecuteType {
        Draw,
        DrawIndexed,
        DrawMeshTask,
        Dispatch,
        None = -1
    };

    void SetExecuteInfo(ExecuteType type,
                        ::std::optional<uint32_t> startIdx = ::std::nullopt,
                        ::std::optional<uint32_t> drawCount = ::std::nullopt);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence& result) override;

    Type_STLString mPipelineName;

    Type_STLVector<::std::pair<Type_STLString, Type_STLString>> mConfigs;
    Type_STLVector<::std::pair<Type_STLString, RenderPassBinding::PushContants>>
        mPushConstants;
    ::std::optional<
        ::std::pair<Type_STLString, RenderPassBinding::BindlessDescBufInfo>>
        mBindlessDesc;

    Buffer* mArgumentBuffer;

    ::std::optional<vk::Rect2D> mRenderArea;
    ::std::optional<vk::Viewport> mViewport;
    ::std::optional<vk::Rect2D> mScissor;

    ExecuteType mExecuteType {ExecuteType::None};
    uint32_t mStartIdx {0};
    uint32_t mDrawCount {0};
};

class CopyPassConfig : public IPassConfig {
    using Self = CopyPassConfig;
    using Type_Region =
        ::std::variant<vk::BufferCopy2, vk::BufferImageCopy2, vk::ImageCopy2>;

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

    Self& SetAsync(bool isAsync);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence& result) override;

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

template <class T>
RenderPassConfig::Self& RenderPassConfig::SetBinding(const char* name,
                                                     T* pPushConstants) {
    mPushConstants.emplace_back(::std::pair {
        name,
        RenderPassBinding::PushContants {sizeof(T), pPushConstants}});
    return *this;
}

}  // namespace IntelliDesign_NS::Vulkan::Core