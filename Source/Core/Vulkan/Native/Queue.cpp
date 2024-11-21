#include "Queue.h"

#include "CommandBuffer.h"

namespace IntelliDesign_NS::Vulkan::Core {

Queue::Queue(Device& device, uint32_t familyIndex,
             vk::QueueFamilyProperties properties, vk::Bool32 canPresent,
             uint32_t index)
    : VulkanResource(VK_NULL_HANDLE, &device),
      mFamilyIndex(familyIndex),
      mIndex(index),
      mCanPresent(canPresent),
      mProperties(properties) {
    SetHandle(GetDevice().GetHandle().getQueue(mFamilyIndex, index));
}

Queue::Queue(Queue&& other) noexcept
    : VulkanResource(std::move(other)),
      mFamilyIndex(::std::exchange(other.mFamilyIndex, {})),
      mIndex(std::exchange(other.mIndex, 0)),
      mCanPresent(std::exchange(other.mCanPresent, false)),
      mProperties(std::exchange(other.mProperties, {})) {}

uint32_t Queue::GetFamilyIndex() const {
    return mFamilyIndex;
}

uint32_t Queue::GetIndex() const {
    return mIndex;
}

const vk::QueueFamilyProperties& Queue::GetProperties() const {
    return mProperties;
}

vk::Bool32 Queue::SupportPresent() const {
    return mCanPresent;
}

void Queue::Submit(
    const CommandBuffer& commandBuffer, vk::Fence fence,
    vk::ArrayProxyNoTemporaries<const vk::SemaphoreSubmitInfo> const& waits,
    vk::ArrayProxyNoTemporaries<const vk::SemaphoreSubmitInfo> const& signals)
    const {
    auto cb = commandBuffer.GetHandle();
    vk::CommandBufferSubmitInfo cbSubmitInfo {cb};
    vk::SubmitInfo2 info {{}, waits, cbSubmitInfo, signals};
    GetHandle().submit2(info, fence);
}

vk::Result Queue::Present(const vk::PresentInfoKHR& info) const {
    if (!mCanPresent) {
        return vk::Result::eErrorIncompatibleDisplayKHR;
    }
    return GetHandle().presentKHR(info);
}

}  // namespace IntelliDesign_NS::Vulkan::Core