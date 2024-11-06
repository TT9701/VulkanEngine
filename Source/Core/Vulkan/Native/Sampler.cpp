#include "Sampler.h"

#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

Sampler::Sampler(Context* context, vk::Filter minFilter,
                             vk::Filter magFilter,
                             vk::SamplerAddressMode addressModeU,
                             vk::SamplerAddressMode addressModeV,
                             vk::SamplerAddressMode addressModeW, float maxLod,
                             bool compareEnable, vk::CompareOp compareOp)
    : pContext(context),
      mSampler(CreateSampler(minFilter, magFilter, addressModeU, addressModeV,
                             addressModeW, maxLod, compareEnable, compareOp)) {}

Sampler::~Sampler() {
    pContext->GetDeviceHandle().destroy(mSampler);
}

vk::Sampler Sampler::CreateSampler(vk::Filter minFilter,
                                         vk::Filter magFilter,
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

}  // namespace IntelliDesign_NS::Vulkan::Core