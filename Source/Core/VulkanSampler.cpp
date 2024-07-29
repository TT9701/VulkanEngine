#include "VulkanSampler.hpp"

#include "VulkanContext.hpp"

VulkanSampler::VulkanSampler(VulkanContext* context, vk::Filter minFilter,
                             vk::Filter             magFilter,
                             vk::SamplerAddressMode addressModeU,
                             vk::SamplerAddressMode addressModeV,
                             vk::SamplerAddressMode addressModeW, float maxLod,
                             bool compareEnable, vk::CompareOp compareOp)
    : pContext(context),
      mSampler(CreateSampler(minFilter, magFilter, addressModeU, addressModeV,
                             addressModeW, maxLod, compareEnable, compareOp)) {}

VulkanSampler::~VulkanSampler() {
    pContext->GetDeviceHandle().destroy(mSampler);
}

vk::Sampler VulkanSampler::CreateSampler(vk::Filter             minFilter,
                                         vk::Filter             magFilter,
                                         vk::SamplerAddressMode addressModeU,
                                         vk::SamplerAddressMode addressModeV,
                                         vk::SamplerAddressMode addressModeW,
                                         float maxLod, bool compareEnable,
                                         vk::CompareOp compareOp) const {
    vk::SamplerCreateInfo info {};
    info.setMinFilter(minFilter)
        .setMagFilter(magFilter)
        .setMipmapMode(maxLod > 0.0f ? vk::SamplerMipmapMode::eLinear
                                     : vk::SamplerMipmapMode::eNearest)
        .setAddressModeU(addressModeU)
        .setAddressModeV(addressModeV)
        .setAddressModeW(addressModeW)
        .setMaxLod(maxLod)
        .setCompareEnable(compareEnable)
        .setCompareOp(compareOp);

    return pContext->GetDeviceHandle().createSampler(info);
}