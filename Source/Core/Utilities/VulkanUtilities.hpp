#pragma once

#include <iostream>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

#include "MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core::Utils {

Type_STLVector<Type_STLString> FilterStringList(
    ::std::span<Type_STLString> available, ::std::span<Type_STLString> request);

void TransitionImageLayout(vk::CommandBuffer cmd, vk::Image img,
                           vk::ImageLayout currentLayout,
                           vk::ImageLayout newLayout);

#define VK_CHECK(x)                                                        \
    {                                                                      \
        vk::Result err = x;                                                \
        if (err != vk::Result::eSuccess) {                                 \
            ::std::cout << "Detected Vulkan error: " << vk::to_string(err) \
                        << "\n";                                           \
            abort();                                                       \
        }                                                                  \
    }

}  // namespace IntelliDesign_NS::Vulkan::Core::Utils

#include "Core/Utilities/Logger.hpp"