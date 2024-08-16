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
        .setSubresourceRange(
            {newLayout == vk::ImageLayout::eDepthAttachmentOptimal
                 ? vk::ImageAspectFlagBits::eDepth
                 : vk::ImageAspectFlagBits::eColor,
             0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers})
        .setImage(img);

    vk::DependencyInfo depInfo {};
    depInfo.setImageMemoryBarrierCount(1u).setImageMemoryBarriers(imgBarrier);

    cmd.pipelineBarrier2(depInfo);
}

}  // namespace Utils
