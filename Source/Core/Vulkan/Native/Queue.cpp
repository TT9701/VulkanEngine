#include "Queue.h"

#include "Device.h"

namespace IntelliDesign_NS::Vulkan::Core {

Queue::Queue(Device& device, uint32_t family_index,
             vk::QueueFamilyProperties const& properties, vk::Bool32 canPresent,
             uint32_t index)
    : mDevice(device),
      mFamilyIndex(family_index),
      mIndex(index),
      mCanPresent(canPresent),
      mProperties(properties) {
    vk::DeviceQueueInfo2 info {};
    info.setQueueFamilyIndex(family_index).setQueueIndex(index);
    mHandle = device.GetHandle().getQueue2(info);
}

vk::Queue const* Queue::operator->() const {
    return &mHandle;
}

vk::Queue Queue::GetHandle() const {
    return mHandle;
}

uint32_t Queue::GetFamilyIndex() const {
    return mFamilyIndex;
}

uint32_t Queue::GetIndex() const {
    return mIndex;
}

vk::QueueFamilyProperties const& Queue::GetFamilyProperties() const {
    return mProperties.queueFamilyProperties;
}

vk::Bool32 Queue::SupportPresent() const {
    return mCanPresent;
}

void Queue::Submit(CommandBuffer const& cmd, vk::Fence) const {
    vk::SubmitInfo2 info {};
    // mHandle.submit2()
}

vk::Result Queue::Present(vk::PresentInfoKHR const& info) const {
    if (!mCanPresent) {
        return vk::Result::eErrorIncompatibleDisplayKHR;
    }

    return mHandle.presentKHR(info);
}

}  // namespace IntelliDesign_NS::Vulkan::Core