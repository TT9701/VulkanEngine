#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
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
    CommandManager(VulkanContext& ctx);

    ~CommandManager() = default;

    CLASS_MOVABLE_ONLY(CommandManager);

public:
    QueueSubmitRequest Submit(vk::CommandBuffer cmd, vk::Queue queue,
                              ::std::span<SemSubmitInfo> waitInfos = {},
                              ::std::span<SemSubmitInfo> signalInfos = {},
                              vk::Fence fence = VK_NULL_HANDLE);

private:
    VulkanContext& mContex;
};

}  // namespace IntelliDesign_NS::Vulkan::Core