#include "VulkanCommandManager.hpp"

#include "Core/Utilities/Logger.hpp"
#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanSyncStructures.hpp"

VulkanCommandManager::VulkanCommandManager(VulkanContext* ctx, uint32_t count,
                                           uint32_t concurrentCommandsCount,
                                           uint32_t queueFamilyIndex,
                                           vk::CommandPoolCreateFlags flags)
    : pContex(ctx),
      mCommandInFlight(concurrentCommandsCount),
      mQueueFamilyIndex(queueFamilyIndex),
      mSPCommandPool(CreateCommandPool(flags)),
      mSPCommandbuffers(CreateCommandBuffers(count)),
      mSPFences(CreateFences()) {
    mIsSubmitted.resize(mCommandInFlight);
}

void VulkanCommandManager::Submit(vk::Queue              queue,
                                  vk::SubmitInfo2 const& submitInfo) {
    pContex->GetDeviceHandle().resetFences(
        mSPFences[mFenceCurrentIndex]->GetHandle());

    queue.submit2(submitInfo, mSPFences[mFenceCurrentIndex]->GetHandle());
    mIsSubmitted[mFenceCurrentIndex] = true;
}

void VulkanCommandManager::GoToNextCmdBuffer() {
    mCommandBufferCurrentIndex =
        (mCommandBufferCurrentIndex + 1) % mSPCommandbuffers->GetBufferCount();
    mFenceCurrentIndex = (mFenceCurrentIndex + 1) % mCommandInFlight;
}

void VulkanCommandManager::WaitUntilSubmitIsComplete() {
    if (!mIsSubmitted[mFenceCurrentIndex])
        return;

    const auto result = pContex->GetDeviceHandle().waitForFences(
        mSPFences[mFenceCurrentIndex]->GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS);

    if (result == vk::Result::eTimeout) {
        ::std::cerr << "Timeout! \n";
        pContex->GetDeviceHandle().waitIdle();
    }

    mIsSubmitted[mFenceCurrentIndex] = false;
}

void VulkanCommandManager::WaitUntilAllSubmitsAreComplete() {
    for (uint32_t index = 0; auto& fence : mSPFences) {
        VK_CHECK(pContex->GetDeviceHandle().waitForFences(
            fence->GetHandle(), vk::True, VulkanFence::TIME_OUT_NANO_SECONDS));
        pContex->GetDeviceHandle().resetFences(fence->GetHandle());
        mIsSubmitted[index++] = false;
    }
}

vk::CommandBuffer VulkanCommandManager::GetCmdBufferToBegin() const {
    VK_CHECK(pContex->GetDeviceHandle().waitForFences(
        mSPFences[mFenceCurrentIndex]->GetHandle(), vk::True,
        VulkanFence::TIME_OUT_NANO_SECONDS));

    auto currentCmdBuf =
        mSPCommandbuffers->GetHandle(mCommandBufferCurrentIndex);

    currentCmdBuf.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    currentCmdBuf.begin(beginInfo);

    return currentCmdBuf;
}

vk::Fence VulkanCommandManager::GetCurrentFence() const {
    return mSPFences[mFenceCurrentIndex]->GetHandle();
}

void VulkanCommandManager::EndCmdBuffer(vk::CommandBuffer cmd) const {
    cmd.end();
}

SharedPtr<VulkanCommandPool> VulkanCommandManager::CreateCommandPool(
    vk::CommandPoolCreateFlags flags) {
    return MakeShared<VulkanCommandPool>(pContex, mQueueFamilyIndex, flags);
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