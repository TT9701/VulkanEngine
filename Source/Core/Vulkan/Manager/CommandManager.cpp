#include "CommandManager.h"

#include "Context.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/SyncStructures.h"

namespace IntelliDesign_NS::Vulkan::Core {

void QueueSubmitRequest::Wait_CPUBlocking() {
    while (*pTimelineValue < mFenceValue) {}
}

CommandManager::CommandManager(VulkanContext* ctx) : pContex(ctx) {}

QueueSubmitRequest CommandManager::Submit(
    vk::CommandBuffer cmd, vk::Queue queue,
    ::std::span<SemSubmitInfo> waitInfos,
    ::std::span<SemSubmitInfo> signalInfos, vk::Fence fence) {
    uint64_t signalValue {0ui64};

    vk::CommandBufferSubmitInfo cmdInfo {cmd};

    Type_STLVector<vk::SemaphoreSubmitInfo> waits {};
    for (auto& info : waitInfos) {
        waits.emplace_back(info.sem, info.value, info.stage);
    }

    Type_STLVector<vk::SemaphoreSubmitInfo> signals {};
    for (auto& info : signalInfos) {
        signals.emplace_back(info.sem, info.value, info.stage);
        signalValue = signalValue > info.value ? signalValue : info.value;
    }

    vk::SubmitInfo2 submit {{}, waits, cmdInfo, signals};

    queue.submit2(submit, fence);

    auto& timelineSem = pContex->GetTimelineSemphore();
    auto timelineValue = timelineSem.GetValue();
    if (signalValue - timelineValue > 0) {
        timelineSem.IncreaseValue(signalValue - timelineValue);
    } else {
        throw ::std::runtime_error("");
    }

    return {signalValue, timelineSem.GetValueAddress()};
}

}  // namespace IntelliDesign_NS::Vulkan::Core