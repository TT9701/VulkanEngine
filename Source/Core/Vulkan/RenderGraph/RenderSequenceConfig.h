#pragma once

#include "RenderSequence.h"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderSequenceConfig {
public:
    RenderPassConfig& AddRenderPass(const char* passName,
                                    const char* pipelineName);

    RenderPassConfig& AddCopyPass(const char* passName);

    void Compile(RenderSequence& result);

private:
    Type_STLVector<RenderPassConfig> mPassConfigs;
};

class RenderPassConfig {
    using Self = RenderPassConfig;

public:
    RenderPassConfig(const char* passName, const char* pipelineName);

    Self& SetBinding(const char* param, const char* argument);

    template <class T>
    Self& SetBinding(T* pPushConstants) {
        mPushConstants =
            RenderPassBinding::PushContants {sizeof(T), pPushConstants};
        return *this;
    }

    Self& SetBinding(const char* param,
                     RenderPassBinding::BindlessDescBufInfo bindless);

    Self& SetRenderArea(vk::Rect2D const& area);
    Self& SetViewport(vk::Viewport const& viewport);
    Self& SetScissor(vk::Rect2D const& scissor);

    friend RenderSequenceConfig;

private:
    void Compile(RenderSequence::RenderPassBindingInfo& info);

    Type_STLString mPassName;
    Type_STLString mPipelineName;

    Type_STLVector<::std::pair<Type_STLString, Type_STLString>> mConfigs;
    ::std::optional<RenderPassBinding::PushContants> mPushConstants;
    ::std::optional<
        ::std::pair<Type_STLString, RenderPassBinding::BindlessDescBufInfo>>
        mBindlessDesc;

    ::std::optional<vk::Rect2D> mRenderArea;
    ::std::optional<vk::Viewport> mViewport;
    ::std::optional<vk::Rect2D> mScissor;
};

}  // namespace IntelliDesign_NS::Vulkan::Core