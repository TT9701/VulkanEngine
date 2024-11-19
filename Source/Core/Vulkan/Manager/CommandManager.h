#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

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
                   uint32_t concurrentCommandsCount,
                   uint32_t gfxQueueFamilyIndex, uint32_t cmpQueueFamilyIndex,
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

    CmdBufferToBegin GetGfxCmdBufToBegin();
    CmdBufferToBegin GetCmpCmdBufToBegin();

    vk::Fence GetCurFence() const;

private:
    SharedPtr<CommandPool> CreateCmdPool(uint32_t queueFamilyIndex,
                                         vk::CommandPoolCreateFlags flags);

    SharedPtr<CommandBuffers> CreateCmdBufs(CommandPool* pool, uint32_t count);

    Type_STLVector<SharedPtr<Fence>> CreateFences();

private:
    Context* pContex;
    friend CmdBufferToBegin;

    uint32_t mCmdInFlight;
    uint32_t mGfxQueueFamilyIndex;
    uint32_t mCmpQueueFamilyIndex;

    SharedPtr<CommandPool> mGfxCmdPool;
    SharedPtr<CommandPool> mCmpCmdPool;

    SharedPtr<CommandBuffers> mGfxCmdbufs;
    SharedPtr<CommandBuffers> mCmpCmdbufs;

    Type_STLVector<SharedPtr<Fence>> mFences;

    Type_STLVector<bool> mIsSubmitted {};
    uint32_t mFenceCurIdx {};
    uint32_t mCmdBufCurIdx {};
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