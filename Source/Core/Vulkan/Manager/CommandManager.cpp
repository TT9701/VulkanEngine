#include "CommandManager.hpp"

#include "Context.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/Vulkan/Native/Commands.hpp"
#include "Core/Vulkan/Native/SyncStructures.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

void QueueSubmitRequest::Wait_CPUBlocking() {
    while (*pTimelineValue < mFenceValue) {}
}

CmdBufferToBegin::CmdBufferToBegin(vk::CommandBuffer cmd,
                                   CommandManager* manager)
    : pManager(manager), mBuffer(cmd) {
    mBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    mBuffer.begin(beginInfo);
}

CmdBufferToBegin::~CmdBufferToBegin() {
    pManager->mCmdBufferCurIdx = (pManager->mCmdBufferCurIdx + 1)
                               % pManager->mSPCommandbuffers->GetBufferCount();
    pManager->mFenceCurIdx =
        (pManager->mFenceCurIdx + 1) % pManager->mCommandInFlight;
}

void CmdBufferToBegin::End() {
    mBuffer.end();
}

CommandManager::CommandManager(Context* ctx, uint32_t count,
                               uint32_t concurrentCommandsCount,
                               uint32_t queueFamilyIndex,
                               vk::CommandPoolCreateFlags flags)
    : pContex(ctx),
      mCommandInFlight(concurrentCommandsCount),
      mGraphicsQueueFamilyIndex(queueFamilyIndex),
      mSPCommandPool(CreateCommandPool(flags)),
      mSPCommandbuffers(CreateCommandBuffers(count)),
      mSPFences(CreateFences()) {
    mIsSubmitted.resize(mCommandInFlight);
}

QueueSubmitRequest CommandManager::Submit(
    vk::CommandBuffer cmd, vk::Queue queue,
    ::std::span<SemSubmitInfo> waitInfos,
    ::std::span<SemSubmitInfo> signalInfos) {
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

    pContex->GetDeviceHandle().resetFences(GetCurrentFence());

    queue.submit2(submit, GetCurrentFence());
    mIsSubmitted[mFenceCurIdx] = true;

    auto timelineValue = pContex->GetTimelineSemphore()->GetValue();
    if (signalValue - timelineValue > 0) {
        pContex->GetTimelineSemphore()->IncreaseValue(signalValue
                                                      - timelineValue);
    } else {
        throw ::std::runtime_error("");
    }

    return {signalValue, pContex->GetTimelineSemphore()->GetValueAddress()};
}

void CommandManager::WaitUntilSubmitIsComplete() {
    if (!mIsSubmitted[mFenceCurIdx])
        return;

    const auto result = pContex->GetDeviceHandle().waitForFences(
        GetCurrentFence(), vk::True, Fence::TIME_OUT_NANO_SECONDS);

    if (result == vk::Result::eTimeout) {
        ::std::cerr << "Timeout! \n";
        pContex->GetDeviceHandle().waitIdle();
    }

    mIsSubmitted[mFenceCurIdx] = false;
}

void CommandManager::WaitUntilAllSubmitsAreComplete() {
    for (uint32_t index = 0; auto& fence : mSPFences) {
        VK_CHECK(pContex->GetDeviceHandle().waitForFences(
            fence->GetHandle(), vk::True, Fence::TIME_OUT_NANO_SECONDS));
        pContex->GetDeviceHandle().resetFences(fence->GetHandle());
        mIsSubmitted[index++] = false;
    }
}

CmdBufferToBegin CommandManager::GetCmdBufferToBegin() {
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        GetCurrentFence(), vk::True, Fence::TIME_OUT_NANO_SECONDS));

    auto currentCmdBuf = mSPCommandbuffers->GetHandle(mCmdBufferCurIdx);

    return {currentCmdBuf, this};
}

vk::Fence CommandManager::GetCurrentFence() const {
    return mSPFences[mFenceCurIdx]->GetHandle();
}

SharedPtr<CommandPool> CommandManager::CreateCommandPool(
    vk::CommandPoolCreateFlags flags) {
    return MakeShared<CommandPool>(pContex, mGraphicsQueueFamilyIndex, flags);
}

SharedPtr<CommandBuffers> CommandManager::CreateCommandBuffers(uint32_t count) {
    return MakeShared<CommandBuffers>(pContex, mSPCommandPool.get(), count);
}

Type_STLVector<SharedPtr<Fence>> CommandManager::CreateFences() {
    Type_STLVector<SharedPtr<Fence>> vec;
    for (uint32_t i = 0; i < mCommandInFlight; ++i) {
        vec.push_back(MakeShared<Fence>(pContex));
    }
    return vec;
}

ImmediateSubmitManager::ImmediateSubmitManager(Context* ctx,
                                               uint32_t queueFamilyIndex)
    : pContex(ctx),
      mQueueFamilyIndex(queueFamilyIndex),
      mSPFence(CreateFence()),
      mSPCommandPool(CreateCommandPool()),
      mSPCommandBuffer(CreateCommandBuffer()) {
    DBG_LOG_INFO("Vulkan Immediate submit CommandPool & CommandBuffer Created");
}

void ImmediateSubmitManager::Submit(
    std::function<void(vk::CommandBuffer cmd)>&& function) const {
    auto func =
        ::std::forward<std::function<void(vk::CommandBuffer cmd)>>(function);

    pContex->GetDeviceHandle().resetFences(mSPFence->GetHandle());

    auto cmd = mSPCommandBuffer->GetHandle();

    cmd.reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);

    func(cmd);

    cmd.end();

    vk::CommandBufferSubmitInfo cmdSubmitInfo {cmd};
    vk::SubmitInfo2 submit {{}, {}, cmdSubmitInfo, {}};

    pContex->GetDevice()->GetGraphicQueue().submit2(submit,
                                                    mSPFence->GetHandle());
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        mSPFence->GetHandle(), vk::True, Fence::TIME_OUT_NANO_SECONDS));
}

SharedPtr<Fence> ImmediateSubmitManager::CreateFence() {
    return MakeShared<Fence>(pContex);
}

SharedPtr<CommandBuffers> ImmediateSubmitManager::CreateCommandBuffer() {
    return MakeShared<CommandBuffers>(pContex, mSPCommandPool.get());
}

SharedPtr<CommandPool> ImmediateSubmitManager::CreateCommandPool() {
    return MakeShared<CommandPool>(pContex, mQueueFamilyIndex);
}

}  // namespace IntelliDesign_NS::Vulkan::Core