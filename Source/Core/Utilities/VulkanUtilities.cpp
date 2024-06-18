#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_EXTERNAL_MEMORY 1
#define VMA_IMPLEMENTATION
#include "VulkanUtilities.hpp"

namespace Utils {

std::vector<std::string> FilterStringList(std::vector<std::string> available,
                                          std::vector<std::string> request) {
    ::std::ranges::sort(available);
    ::std::ranges::sort(request);
    ::std::vector<::std::string> result {};
    ::std::ranges::set_intersection(available, request,
                                    ::std::back_inserter(result));
    return result;
}

vk::ImageSubresourceRange GetDefaultImageSubresourceRange(
    vk::ImageAspectFlags flags) {
    return {flags, 0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers};
}

void TransitionImageLayout(vk::CommandBuffer cmd, vk::Image img,
                           vk::ImageLayout currentLayout,
                           vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier2 imgBarrier {};
    imgBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite |
                          vk::AccessFlagBits2::eMemoryRead)
        .setOldLayout(currentLayout)
        .setNewLayout(newLayout)
        .setSubresourceRange(GetDefaultImageSubresourceRange(
            newLayout == vk::ImageLayout::eDepthAttachmentOptimal
                ? vk::ImageAspectFlagBits::eDepth
                : vk::ImageAspectFlagBits::eColor))
        .setImage(img);

    vk::DependencyInfo depInfo {};
    depInfo.setImageMemoryBarrierCount(1u).setImageMemoryBarriers(imgBarrier);

    cmd.pipelineBarrier2(depInfo);
}

vk::SemaphoreSubmitInfo GetDefaultSemaphoreSubmitInfo(
    vk::PipelineStageFlagBits2 stageMask, vk::Semaphore semaphore) {
    vk::SemaphoreSubmitInfo submitInfo {};
    submitInfo.setSemaphore(semaphore)
        .setStageMask(stageMask)
        .setDeviceIndex(0u)
        .setValue(1ui64);

    return submitInfo;
}

vk::CommandBufferSubmitInfo GetDefaultCommandBufferSubmitInfo(
    vk::CommandBuffer cmd) {
    vk::CommandBufferSubmitInfo info {};
    info.setCommandBuffer(cmd);

    return info;
}

vk::SubmitInfo2 SubmitInfo(
    vk::ArrayProxy<vk::CommandBufferSubmitInfo> const& cmd,
    vk::ArrayProxy<vk::SemaphoreSubmitInfo> const&     signalSemaphoreInfo,
    vk::ArrayProxy<vk::SemaphoreSubmitInfo> const&     waitSemaphoreInfo) {
    vk::SubmitInfo2 info = {};
    info.setWaitSemaphoreInfos(waitSemaphoreInfo)
        .setSignalSemaphoreInfos(signalSemaphoreInfo)
        .setCommandBufferInfos(cmd);

    return info;
}

bool LoadShaderModule(const std::string& filePath, vk::Device device,
                      vk::ShaderModule* outShaderModule) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    size_t fileSize = (size_t)file.tellg();

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(buffer);

    vk::ShaderModule shaderModule = device.createShaderModule(createInfo);

    *outShaderModule = shaderModule;
    return true;
}

}  // namespace Utils
