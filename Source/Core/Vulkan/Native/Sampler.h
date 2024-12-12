#pragma once

#include <Vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

class Sampler {
public:
    Sampler(VulkanContext* context, vk::Filter minFilter,
                  vk::Filter magFilter, vk::SamplerAddressMode addressModeU,
                  vk::SamplerAddressMode addressModeV,
                  vk::SamplerAddressMode addressModeW, float maxLod,
                  bool compareEnable, vk::CompareOp compareOp);
    ~Sampler();
    CLASS_MOVABLE_ONLY(Sampler);

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
    VulkanContext* pContext;
    vk::Sampler mSampler;
};

}  // namespace IntelliDesign_NS::Vulkan::Core