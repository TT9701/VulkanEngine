#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Fence;
class CommandBuffers;
class CommandPool;

struct SemSubmitInfo {
    vk::PipelineStageFlagBits2 stage;
    vk::Semaphore sem;
    uint64_t value {0};
};

struct QueueSubmitRequest {
    QueueSubmitRequest(uint64_t value, uint64_t const* timelineValue)
        : mFenceValue(value), pTimelineValue(timelineValue) {}

    void Wait_CPUBlocking();

    uint64_t mFenceValue {0};

private:
    const uint64_t* pTimelineValue;
};

class CommandManager;

struct CmdBufferToBegin {
    CmdBufferToBegin(vk::CommandBuffer cmd, CommandManager* manager);
    ~CmdBufferToBegin();

    vk::CommandBuffer GetHandle() const { return mBuffer; }

    void End();

private:
    CommandManager* pManager;
    vk::CommandBuffer mBuffer;
};

class CommandManager {
public:
    CommandManager(Context* ctx, uint32_t count,
                   uint32_t concurrentCommandsCount, uint32_t queueFamilyIndex,
                   vk::CommandPoolCreateFlags flags =
                       vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    ~CommandManager() = default;

    MOVABLE_ONLY(CommandManager);

public:
    QueueSubmitRequest Submit(vk::CommandBuffer cmd, vk::Queue queue,
                              ::std::span<SemSubmitInfo> waitInfos = {},
                              ::std::span<SemSubmitInfo> signalInfos = {});

    void WaitUntilSubmitIsComplete();

    void WaitUntilAllSubmitsAreComplete();

    CmdBufferToBegin GetCmdBufferToBegin();

    vk::Fence GetCurrentFence() const;

private:
    SharedPtr<CommandPool> CreateCommandPool(vk::CommandPoolCreateFlags flags);

    SharedPtr<CommandBuffers> CreateCommandBuffers(uint32_t count);

    Type_STLVector<SharedPtr<Fence>> CreateFences();

private:
    Context* pContex;
    friend CmdBufferToBegin;

    uint32_t mCommandInFlight;
    uint32_t mGraphicsQueueFamilyIndex;

    SharedPtr<CommandPool> mSPCommandPool;
    SharedPtr<CommandBuffers> mSPCommandbuffers;
    Type_STLVector<SharedPtr<Fence>> mSPFences;

    Type_STLVector<bool> mIsSubmitted {};
    uint32_t mFenceCurIdx {};
    uint32_t mCmdBufferCurIdx {};
};

class ImmediateSubmitManager {
public:
    ImmediateSubmitManager(Context* ctx, uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager() = default;
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    SharedPtr<Fence> CreateFence();
    SharedPtr<CommandBuffers> CreateCommandBuffer();
    SharedPtr<CommandPool> CreateCommandPool();

private:
    Context* pContex;
    uint32_t mQueueFamilyIndex;

    SharedPtr<Fence> mSPFence;
    SharedPtr<CommandPool> mSPCommandPool;
    SharedPtr<CommandBuffers> mSPCommandBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core