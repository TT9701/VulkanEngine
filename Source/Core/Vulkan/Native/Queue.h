#pragma once

#include "VulkanResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Device;
class CommandBuffer;

class Queue final : public VulkanResource<vk::Queue> {
public:
    Queue(Device& device, uint32_t familyIndex,
          vk::QueueFamilyProperties properties, vk::Bool32 canPresent,
          uint32_t index);

    Queue(const Queue&) = delete;
    Queue(Queue&& other) noexcept;
    Queue& operator=(const Queue&) = delete;
    Queue& operator=(Queue&&) = delete;

    uint32_t GetFamilyIndex() const;

    uint32_t GetIndex() const;

    const vk::QueueFamilyProperties& GetProperties() const;

    vk::Bool32 SupportPresent() const;

    void Submit(
        const CommandBuffer& commandBuffer, vk::Fence fence,
        vk::ArrayProxyNoTemporaries<const vk::SemaphoreSubmitInfo> const&
            waits = {},
        vk::ArrayProxyNoTemporaries<const vk::SemaphoreSubmitInfo> const&
            signals = {}) const;

    vk::Result Present(const vk::PresentInfoKHR& info) const;

private:
    uint32_t mFamilyIndex {0};
    uint32_t mIndex {0};
    vk::Bool32 mCanPresent = false;
    vk::QueueFamilyProperties mProperties {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core