#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core::Utils {

::std::vector<::std::string> FilterStringList(
    ::std::span<::std::string> available, ::std::span<::std::string> request);

void TransitionImageLayout(vk::CommandBuffer cmd, vk::Image img,
                           vk::ImageLayout currentLayout,
                           vk::ImageLayout newLayout);

}  // namespace Utils