#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_EXTERNAL_MEMORY 1
#define VMA_IMPLEMENTATION
#include "VulkanUtilities.hpp"

namespace Utils {

std::vector<std::string> FilterStringList(std::span<std::string> available,
                                          std::span<std::string> request) {
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
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite
                          | vk::AccessFlagBits2::eMemoryRead)
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
    vk::PipelineStageFlagBits2 stageMask, vk::Semaphore semaphore,
    uint64_t value) {
    vk::SemaphoreSubmitInfo submitInfo {};
    submitInfo.setSemaphore(semaphore)
        .setStageMask(stageMask)
        .setDeviceIndex(0u)
        .setValue(value);

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

}  // namespace Utils
