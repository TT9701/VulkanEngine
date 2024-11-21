#include "FencePool.h"

#include "Device.h"

namespace IntelliDesign_NS::Vulkan::Core {
FencePool::FencePool(Device& device) : mDevice {device} {}

FencePool::~FencePool() {
    Wait();
    Reset();

    for (VkFence fence : mFences) {
        vkDestroyFence(mDevice.GetHandle(), fence, nullptr);
    }

    mFences.clear();
}

vk::Fence FencePool::RequestFence(vk::FenceCreateFlags flags) {
    if (mActiveFenceCount < mFences.size()) {
        return mFences[mActiveFenceCount++];
    }

    vk::FenceCreateInfo info {flags};

    auto fence = mDevice.GetHandle().createFence(info);

    mFences.push_back(fence);

    mActiveFenceCount++;

    return mFences.back();
}

vk::Result FencePool::Wait(uint32_t timeout) const {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }
    return mDevice.GetHandle().waitForFences(mFences, vk::True, timeout);
}

vk::Result FencePool::Reset() {
    if (mActiveFenceCount < 1 || mFences.empty()) {
        return vk::Result::eSuccess;
    }

    mDevice.GetHandle().resetFences(mFences);

    mActiveFenceCount = 0;

    return vk::Result::eSuccess;
}
}  // namespace IntelliDesign_NS::Vulkan::Core