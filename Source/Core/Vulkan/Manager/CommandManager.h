#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Fence;
class CommandBuffer;
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

class CommandManager {
public:
    CommandManager(Context* ctx);

    ~CommandManager() = default;

    MOVABLE_ONLY(CommandManager);

public:
    /**
     * .
     * 
     * \param cmd
     * \param queue
     * \param waitInfos
     * \param signalInfos
     * \param fence
     * \return 
     */
    QueueSubmitRequest Submit(vk::CommandBuffer cmd, vk::Queue queue,
                              ::std::span<SemSubmitInfo> waitInfos = {},
                              ::std::span<SemSubmitInfo> signalInfos = {}, 
                              vk::Fence fence = VK_NULL_HANDLE);

private:
    Context* pContex;
};

class ImmediateSubmitManager {
public:
    ImmediateSubmitManager(Context* ctx, uint32_t queueFamilyIndex);
    ~ImmediateSubmitManager();
    MOVABLE_ONLY(ImmediateSubmitManager);

public:
    void Submit(::std::function<void(vk::CommandBuffer cmd)>&& function) const;

private:
    vk::Fence CreateFence();
    SharedPtr<CommandBuffer> CreateCommandBuffer();
    SharedPtr<CommandPool> CreateCommandPool();

private:
    Context* pContex;
    uint32_t mQueueFamilyIndex;

    vk::Fence mFence;
    SharedPtr<CommandPool> mSPCommandPool;
    SharedPtr<CommandBuffer> mSPCommandBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core