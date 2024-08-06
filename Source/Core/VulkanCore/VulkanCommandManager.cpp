#include "VulkanCommandManager.hpp"

#include "Core/Utilities/Logger.hpp"
#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanSyncStructures.hpp"

void VulkanQueueSubmitRequest::Wait_CPUBlocking() {
    while (*pTimelineValue < mFenceValue) {}
}

CmdBufferToBegin::CmdBufferToBegin(vk::CommandBuffer     cmd,
                                   VulkanCommandManager* manager)
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

VulkanCommandManager::VulkanCommandManager(VulkanContext* ctx, uint32_t count,
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

VulkanQueueSubmitRequest VulkanCommandManager::Submit(
    vk::CommandBuffer cmd, vk::Queue queue,
    ::std::span<SemSubmitInfo> waitInfos,
    ::std::span<SemSubmitInfo> signalInfos) {
    uint64_t signalValue {0ui64};

    auto cmdInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);

    ::std::vector<vk::SemaphoreSubmitInfo> waits {};
    for (auto& info : waitInfos) {
        waits.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
            info.flags, info.sem, info.value));
    }

    ::std::vector<vk::SemaphoreSubmitInfo> signals {};
    for (auto& info : signalInfos) {
        signals.push_back(Utils::GetDefaultSemaphoreSubmitInfo(
            info.flags, info.sem, info.value));
        signalValue = signalValue > info.value ? signalValue : info.value;
    }

    auto submit = Utils::SubmitInfo(cmdInfo, signals, waits);

    pContex->GetDeviceHandle().resetFences(
        mSPFences[mFenceCurIdx]->GetHandle());

    queue.submit2(submit, mSPFences[mFenceCurIdx]->GetHandle());
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

void VulkanCommandManager::WaitUntilSubmitIsComplete() {
    if (!mIsSubmitted[mFenceCurIdx])
        return;

    const auto result = pContex->GetDeviceHandle().waitForFences(
        mSPFences[mFenceCurIdx]->GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS);

    if (result == vk::Result::eTimeout) {
        ::std::cerr << "Timeout! \n";
        pContex->GetDeviceHandle().waitIdle();
    }

    mIsSubmitted[mFenceCurIdx] = false;
}

void VulkanCommandManager::WaitUntilAllSubmitsAreComplete() {
    for (uint32_t index = 0; auto& fence : mSPFences) {
        VK_CHECK(pContex->GetDeviceHandle().waitForFences(
            fence->GetHandle(), vk::True, VulkanFence::TIME_OUT_NANO_SECONDS));
        pContex->GetDeviceHandle().resetFences(fence->GetHandle());
        mIsSubmitted[index++] = false;
    }
}

CmdBufferToBegin VulkanCommandManager::GetCmdBufferToBegin() {
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        mSPFences[mFenceCurIdx]->GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS));

    auto currentCmdBuf = mSPCommandbuffers->GetHandle(mCmdBufferCurIdx);

    return {currentCmdBuf, this};
}

vk::Fence VulkanCommandManager::GetCurrentFence() const {
    return mSPFences[mFenceCurIdx]->GetHandle();
}

SharedPtr<VulkanCommandPool> VulkanCommandManager::CreateCommandPool(
    vk::CommandPoolCreateFlags flags) {
    return MakeShared<VulkanCommandPool>(pContex, mGraphicsQueueFamilyIndex,
                                         flags);
}

SharedPtr<VulkanCommandBuffers> VulkanCommandManager::CreateCommandBuffers(
    uint32_t count) {
    return MakeShared<VulkanCommandBuffers>(pContex, mSPCommandPool.get(),
                                            count);
}

std::vector<SharedPtr<VulkanFence>> VulkanCommandManager::CreateFences() {
    ::std::vector<SharedPtr<VulkanFence>> vec;
    for (uint32_t i = 0; i < mCommandInFlight; ++i) {
        vec.push_back(MakeShared<VulkanFence>(pContex));
    }
    return vec;
}

ImmediateSubmitManager::ImmediateSubmitManager(VulkanContext* ctx,
                                               uint32_t       queueFamilyIndex)
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

    auto cmdSubmitInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);
    auto submit        = Utils::SubmitInfo(cmdSubmitInfo, {}, {});

    pContex->GetDevice()->GetGraphicQueue().submit2(submit,
                                                    mSPFence->GetHandle());
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        mSPFence->GetHandle(), vk::True, VulkanFence::TIME_OUT_NANO_SECONDS));
}

SharedPtr<VulkanFence> ImmediateSubmitManager::CreateFence() {
    return MakeShared<VulkanFence>(pContex);
}

SharedPtr<VulkanCommandBuffers> ImmediateSubmitManager::CreateCommandBuffer() {
    return MakeShared<VulkanCommandBuffers>(pContex, mSPCommandPool.get());
}

SharedPtr<VulkanCommandPool> ImmediateSubmitManager::CreateCommandPool() {
    return MakeShared<VulkanCommandPool>(pContex, mQueueFamilyIndex);
}