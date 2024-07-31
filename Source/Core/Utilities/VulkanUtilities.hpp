#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace Utils {

::std::vector<::std::string> FilterStringList(
    ::std::span<::std::string> available, ::std::span<::std::string> request);

vk::ImageSubresourceRange GetDefaultImageSubresourceRange(
    vk::ImageAspectFlags flags);

void TransitionImageLayout(vk::CommandBuffer cmd, vk::Image img,
                           vk::ImageLayout currentLayout,
                           vk::ImageLayout newLayout);

vk::SemaphoreSubmitInfo GetDefaultSemaphoreSubmitInfo(
    vk::PipelineStageFlagBits2 stageMask, vk::Semaphore semaphore);

vk::CommandBufferSubmitInfo GetDefaultCommandBufferSubmitInfo(
    vk::CommandBuffer cmd);

vk::SubmitInfo2 SubmitInfo(
    vk::ArrayProxy<vk::CommandBufferSubmitInfo> const& cmd,
    vk::ArrayProxy<vk::SemaphoreSubmitInfo> const&     signalSemaphoreInfo,
    vk::ArrayProxy<vk::SemaphoreSubmitInfo> const&     waitSemaphoreInfo);
}  // namespace Utils