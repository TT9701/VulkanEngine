#pragma once

#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {

class Device;

class FencePool {
public:
    FencePool(Device& device);

    FencePool(const FencePool&) = delete;

    FencePool(FencePool&& other) = delete;

    ~FencePool();

    FencePool& operator=(const FencePool&) = delete;

    FencePool& operator=(FencePool&&) = delete;

    vk::Fence RequestFence(
        vk::FenceCreateFlags flags = vk::FenceCreateFlagBits::eSignaled);

    vk::Result Wait(
        uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;

    vk::Result Reset();

private:
    Device& mDevice;

    std::vector<vk::Fence> mFences;

    uint32_t mActiveFenceCount {0};
};

}  // namespace IntelliDesign_NS::Vulkan::Core