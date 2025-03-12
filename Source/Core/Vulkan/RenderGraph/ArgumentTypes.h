#pragma once

#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {
namespace RenderPassBinding {

struct RenderInfo {
    vk::Rect2D renderArea;
    uint32_t layerCount;
    uint32_t viewMask;
};

struct BindlessDescBufInfo {
    vk::DeviceAddress deviceAddress;
    vk::DeviceSize offset;
};

}  // namespace RenderPassBinding

}  // namespace IntelliDesign_NS::Vulkan::Core