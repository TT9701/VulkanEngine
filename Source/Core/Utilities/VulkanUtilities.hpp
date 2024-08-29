#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core::Utils {

Type_STLVector<Type_STLString> FilterStringList(
    ::std::span<Type_STLString> available, ::std::span<Type_STLString> request);

void TransitionImageLayout(vk::CommandBuffer cmd, vk::Image img,
                           vk::ImageLayout currentLayout,
                           vk::ImageLayout newLayout);

}  // namespace Utils