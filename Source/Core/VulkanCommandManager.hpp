#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Utilities/Defines.hpp"

class VulkanContext;
class VulkanFence;
class VulkanCommandBuffer;
class VulkanCommandPool;

class VulkanCommandManager {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:

private:

};

class ImmediateSubmitManager {
    USING_TEMPLATE_PTR_TYPE(Type_PInstance, Type_SPInstance);

public:
    ImmediateSubmitManager(Type_SPInstance<VulkanContext> const& ctx,
                           uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    Type_SPInstance<VulkanFence> CreateFence();
    Type_SPInstance<VulkanCommandBuffer> CreateCommandBuffer();
    Type_SPInstance<VulkanCommandPool> CreateCommandPool();

private:
    Type_SPInstance<VulkanContext> pContex;
    uint32_t mQueueFamilyIndex;

    Type_SPInstance<VulkanFence> mSPFence;
    Type_SPInstance<VulkanCommandPool> mSPCommandPool;
    Type_SPInstance<VulkanCommandBuffer> mSPCommandBuffer;
};
