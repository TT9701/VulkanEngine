#include "CommandManager.h"

#include "Context.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/SyncStructures.h"

namespace IntelliDesign_NS::Vulkan::Core {

void QueueSubmitRequest::Wait_CPUBlocking() {
    while (*pTimelineValue < mFenceValue) {}
}

CommandManager::CommandManager(Context* ctx) : pContex(ctx) {}

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

ImmediateSubmitManager::ImmediateSubmitManager(Context* ctx,
                                               uint32_t queueFamilyIndex)
    : pContex(ctx),
      mQueueFamilyIndex(queueFamilyIndex),
      mFence(CreateFence()),
      mSPCommandPool(CreateCommandPool()),
      mSPCommandBuffer(CreateCommandBuffer()) {
    DBG_LOG_INFO("Vulkan Immediate submit CommandPool & CommandBuffer Created");
}

ImmediateSubmitManager::~ImmediateSubmitManager() {
    pContex->GetDevice()->destroy(mFence);
}

void ImmediateSubmitManager::Submit(
    std::function<void(vk::CommandBuffer cmd)>&& function) const {
    auto func =
        ::std::forward<std::function<void(vk::CommandBuffer cmd)>>(function);

    pContex->GetDevice()->resetFences(mFence);

    auto cmd = mSPCommandBuffer->GetHandle();

    cmd.reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);

    func(cmd);

    cmd.end();

    vk::CommandBufferSubmitInfo cmdSubmitInfo {cmd};
    vk::SubmitInfo2 submit {{}, {}, cmdSubmitInfo, {}};

    pContex->GetGraphicsQueue().GetHandle().submit2(submit, mFence);
    VK_CHECK(pContex->GetDevice()->waitForFences(
        mFence, vk::True, FencePool::TIME_OUT_NANO_SECONDS));
}

vk::Fence ImmediateSubmitManager::CreateFence() {
    vk::FenceCreateInfo info {vk::FenceCreateFlagBits::eSignaled};
    return pContex->GetDevice()->createFence(info);
}

SharedPtr<CommandBuffer> ImmediateSubmitManager::CreateCommandBuffer() {
    return MakeShared<CommandBuffer>(pContex, mSPCommandPool.get());
}

SharedPtr<CommandPool> ImmediateSubmitManager::CreateCommandPool() {
    return MakeShared<CommandPool>(pContex, mQueueFamilyIndex);
}

}  // namespace IntelliDesign_NS::Vulkan::Core