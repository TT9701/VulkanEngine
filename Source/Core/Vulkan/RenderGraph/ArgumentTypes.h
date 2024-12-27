#pragma once

#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {
namespace RenderPassBinding {

struct PushContants {
    uint32_t size;
    void* pData;
};

struct RenderInfo {
    vk::Rect2D renderArea;
    uint32_t layerCount;
    uint32_t viewMask;
};

struct BindlessDescBufInfo {
    vk::DeviceAddress deviceAddress;
    vk::DeviceSize offset;
};

struct ArgumentBufferInfo {
    vk::Buffer buffer;
    uint32_t startIndex;
    uint32_t count;
};

}  // namespace RenderPassBinding
}  // namespace IntelliDesign_NS::Vulkan::Core