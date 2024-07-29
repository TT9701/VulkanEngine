#pragma once

#include <Vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

class VulkanContext;

class VulkanSampler {
public:
    VulkanSampler(
        VulkanContext* context, vk::Filter minFilter, vk::Filter magFilter,
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat,
        float maxLod = 0.0f, bool compareEnable = false,
        vk::CompareOp compareOp = vk::CompareOp::eNever);
    ~VulkanSampler();
    MOVABLE_ONLY(VulkanSampler);

public:
    vk::Sampler GetHandle() const { return mSampler; }

private:
    vk::Sampler CreateSampler(
        vk::Filter minFilter, vk::Filter magFilter,
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat,
        float maxLod = 0.0f, bool compareEnable = false,
        vk::CompareOp compareOp = vk::CompareOp::eNever) const;

private:
    VulkanContext* pContext;
    vk::Sampler    mSampler;
};