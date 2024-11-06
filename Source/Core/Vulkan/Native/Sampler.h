#pragma once

#include <Vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class Sampler {
public:
    Sampler(Context* context, vk::Filter minFilter,
                  vk::Filter magFilter, vk::SamplerAddressMode addressModeU,
                  vk::SamplerAddressMode addressModeV,
                  vk::SamplerAddressMode addressModeW, float maxLod,
                  bool compareEnable, vk::CompareOp compareOp);
    ~Sampler();
    MOVABLE_ONLY(Sampler);

public:
    vk::Sampler GetHandle() const { return mSampler; }

private:
    vk::Sampler CreateSampler(vk::Filter minFilter, vk::Filter magFilter,
                              vk::SamplerAddressMode addressModeU,
                              vk::SamplerAddressMode addressModeV,
                              vk::SamplerAddressMode addressModeW, float maxLod,
                              bool compareEnable,
                              vk::CompareOp compareOp) const;

private:
    Context* pContext;
    vk::Sampler mSampler;
};

}  // namespace IntelliDesign_NS::Vulkan::Core