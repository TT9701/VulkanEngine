#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/Defines.hpp"

class VulkanContext;
class VulkanFence;
class VulkanCommandBuffer;
class VulkanCommandPool;

class VulkanCommandManager {
public:

private:

};

class ImmediateSubmitManager {
public:
    ImmediateSubmitManager(SharedPtr<VulkanContext> const& ctx,
                           uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    SharedPtr<VulkanFence> CreateFence();
    SharedPtr<VulkanCommandBuffer> CreateCommandBuffer();
    SharedPtr<VulkanCommandPool> CreateCommandPool();

private:
    SharedPtr<VulkanContext> pContex;
    uint32_t mQueueFamilyIndex;

    SharedPtr<VulkanFence> mSPFence;
    SharedPtr<VulkanCommandPool> mSPCommandPool;
    SharedPtr<VulkanCommandBuffer> mSPCommandBuffer;
};
