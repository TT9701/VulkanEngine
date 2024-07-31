#pragma once

#include <Vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanSampler {
public:
    VulkanSampler(VulkanContext* context, vk::Filter minFilter,
                  vk::Filter magFilter, vk::SamplerAddressMode addressModeU,
                  vk::SamplerAddressMode addressModeV,
                  vk::SamplerAddressMode addressModeW, float maxLod,
                  bool compareEnable, vk::CompareOp compareOp);
    ~VulkanSampler();
    MOVABLE_ONLY(VulkanSampler);

public:
    vk::Sampler GetHandle() const { return mSampler; }

private:
    vk::Sampler CreateSampler(vk::Filter minFilter, vk::Filter magFilter,
                              vk::SamplerAddressMode addressModeU,
                              vk::SamplerAddressMode addressModeV,
                              vk::SamplerAddressMode addressModeW, float maxLod,
                              bool          compareEnable,
                              vk::CompareOp compareOp) const;

private:
    VulkanContext* pContext;
    vk::Sampler    mSampler;
};