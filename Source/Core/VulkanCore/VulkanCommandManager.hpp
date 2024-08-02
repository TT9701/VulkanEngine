#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanContext;
class VulkanFence;
class VulkanCommandBuffers;
class VulkanCommandPool;

namespace VulkanCore::_Detail {

struct SemSubmitInfo {
    vk::PipelineStageFlagBits2 flags;
    vk::Semaphore              sem;
    uint64_t                   value {0};
};

}  // namespace VulkanCore::_Detail

class VulkanCommandManager {
public:
    VulkanCommandManager(
        VulkanContext* ctx, uint32_t count, uint32_t concurrentCommandsCount,
        uint32_t                   queueFamilyIndex,
        vk::CommandPoolCreateFlags flags =
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    ~VulkanCommandManager() = default;

    MOVABLE_ONLY(VulkanCommandManager);

    using Type_SemSubmitInfo =
        ::std::tuple<vk::PipelineStageFlagBits2, vk::Semaphore, uint64_t>;

public:
    void Submit(
        vk::CommandBuffer cmd, vk::Queue queue,
        ::std::vector<VulkanCore::_Detail::SemSubmitInfo> const& waitInfos,
        ::std::vector<VulkanCore::_Detail::SemSubmitInfo> const& signalInfos);

    void GoToNextCmdBuffer();

    void WaitUntilSubmitIsComplete();

    void WaitUntilAllSubmitsAreComplete();

    vk::CommandBuffer GetCmdBufferToBegin() const;

    vk::Fence GetCurrentFence() const;

    void EndCmdBuffer(vk::CommandBuffer cmd) const;

private:
    SharedPtr<VulkanCommandPool> CreateCommandPool(
        vk::CommandPoolCreateFlags flags);

    SharedPtr<VulkanCommandBuffers> CreateCommandBuffers(uint32_t count);

    ::std::vector<SharedPtr<VulkanFence>> CreateFences();

private:
    VulkanContext* pContex;

    uint32_t mCommandInFlight;
    uint32_t mGraphicsQueueFamilyIndex;

    SharedPtr<VulkanCommandPool>          mSPCommandPool;
    SharedPtr<VulkanCommandBuffers>       mSPCommandbuffers;
    ::std::vector<SharedPtr<VulkanFence>> mSPFences;

    ::std::vector<bool> mIsSubmitted {};
    uint32_t            mFenceCurIdx {};
    uint32_t            mCmdBufferCurIdx {};
};

class ImmediateSubmitManager {
public:
    ImmediateSubmitManager(VulkanContext* ctx, uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    SharedPtr<VulkanFence>          CreateFence();
    SharedPtr<VulkanCommandBuffers> CreateCommandBuffer();
    SharedPtr<VulkanCommandPool>    CreateCommandPool();

private:
    VulkanContext* pContex;
    uint32_t       mQueueFamilyIndex;

    SharedPtr<VulkanFence>          mSPFence;
    SharedPtr<VulkanCommandPool>    mSPCommandPool;
    SharedPtr<VulkanCommandBuffers> mSPCommandBuffer;
};
