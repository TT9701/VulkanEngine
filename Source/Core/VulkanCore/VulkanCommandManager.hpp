#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanContext;
class VulkanFence;
class VulkanCommandBuffers;
class VulkanCommandPool;

struct SemSubmitInfo {
    vk::PipelineStageFlagBits2 stage;
    vk::Semaphore sem;
    uint64_t value {0};
};

struct VulkanQueueSubmitRequest {
    VulkanQueueSubmitRequest(uint64_t value, uint64_t const* timelineValue)
        : mFenceValue(value), pTimelineValue(timelineValue) {}

    void Wait_CPUBlocking();

    uint64_t mFenceValue {0};

private:
    const uint64_t* pTimelineValue;
};

class VulkanCommandManager;

struct CmdBufferToBegin {
    CmdBufferToBegin(vk::CommandBuffer cmd, VulkanCommandManager* manager);
    ~CmdBufferToBegin();

    vk::CommandBuffer GetHandle() const { return mBuffer; }

    void End();

private:
    VulkanCommandManager* pManager;
    vk::CommandBuffer mBuffer;
};

class VulkanCommandManager {
public:
    VulkanCommandManager(
        VulkanContext* ctx, uint32_t count, uint32_t concurrentCommandsCount,
        uint32_t queueFamilyIndex,
        vk::CommandPoolCreateFlags flags =
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    ~VulkanCommandManager() = default;

    MOVABLE_ONLY(VulkanCommandManager);

public:
    VulkanQueueSubmitRequest Submit(
        vk::CommandBuffer cmd, vk::Queue queue,
        ::std::span<SemSubmitInfo> waitInfos = {},
        ::std::span<SemSubmitInfo> signalInfos = {});

    void WaitUntilSubmitIsComplete();

    void WaitUntilAllSubmitsAreComplete();

    CmdBufferToBegin GetCmdBufferToBegin();

    vk::Fence GetCurrentFence() const;

private:
    SharedPtr<VulkanCommandPool> CreateCommandPool(
        vk::CommandPoolCreateFlags flags);

    SharedPtr<VulkanCommandBuffers> CreateCommandBuffers(uint32_t count);

    ::std::vector<SharedPtr<VulkanFence>> CreateFences();

private:
    VulkanContext* pContex;
    friend CmdBufferToBegin;

    uint32_t mCommandInFlight;
    uint32_t mGraphicsQueueFamilyIndex;

    SharedPtr<VulkanCommandPool> mSPCommandPool;
    SharedPtr<VulkanCommandBuffers> mSPCommandbuffers;
    ::std::vector<SharedPtr<VulkanFence>> mSPFences;

    ::std::vector<bool> mIsSubmitted {};
    uint32_t mFenceCurIdx {};
    uint32_t mCmdBufferCurIdx {};
};

class ImmediateSubmitManager {
public:
    ImmediateSubmitManager(VulkanContext* ctx, uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    SharedPtr<VulkanFence> CreateFence();
    SharedPtr<VulkanCommandBuffers> CreateCommandBuffer();
    SharedPtr<VulkanCommandPool> CreateCommandPool();

private:
    VulkanContext* pContex;
    uint32_t mQueueFamilyIndex;

    SharedPtr<VulkanFence> mSPFence;
    SharedPtr<VulkanCommandPool> mSPCommandPool;
    SharedPtr<VulkanCommandBuffers> mSPCommandBuffer;
};
