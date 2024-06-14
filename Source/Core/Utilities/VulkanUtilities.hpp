#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace Utils {

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }

        deletors.clear();
    }
};

::std::vector<::std::string> FilterStringList(
    ::std::vector<::std::string> available,
    ::std::vector<::std::string> request);

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

bool LoadShaderModule(const ::std::string& filePath, vk::Device device,
                      vk::ShaderModule* outShaderModule);

}  // namespace Utils