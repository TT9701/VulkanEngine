#include "VulkanCommandManager.hpp"

#include "Core/Utilities/Logger.hpp"
#include "VulkanCommands.hpp"
#include "VulkanContext.hpp"
#include "VulkanSyncStructures.hpp"

using IntelliDesign_NS::Core::MemoryPool::New_Shared;

ImmediateSubmitManager::ImmediateSubmitManager(
    SharedPtr<VulkanContext> const& ctx, uint32_t queueFamilyIndex)
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

    pContex->GetDevice()->GetHandle().resetFences(mSPFence->GetHandle());

    auto cmd = mSPCommandBuffer->GetHandle();

    cmd.reset();

    vk::CommandBufferBeginInfo beginInfo {};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);

    func(cmd);

    cmd.end();

    auto cmdSubmitInfo = Utils::GetDefaultCommandBufferSubmitInfo(cmd);
    auto submit = Utils::SubmitInfo(cmdSubmitInfo, {}, {});

    pContex->GetDevice()->GetGraphicQueues()[0].submit2(submit,
                                                        mSPFence->GetHandle());
    VK_CHECK(pContex->GetDevice()->GetHandle().waitForFences(
        mSPFence->GetHandle(), vk::True, VulkanFence::TIME_OUT_NANO_SECONDS));
}

SharedPtr<VulkanFence> ImmediateSubmitManager::CreateFence() {
    return New_Shared<VulkanFence>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), pContex);
}

SharedPtr<VulkanCommandBuffer>
ImmediateSubmitManager::CreateCommandBuffer() {
    return New_Shared<VulkanCommandBuffer>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), pContex,
        mSPCommandPool);
}

SharedPtr<VulkanCommandPool> ImmediateSubmitManager::CreateCommandPool() {
    return New_Shared<VulkanCommandPool>(
        MemoryPoolInstance::Get()->GetMemPoolResource(), pContex,
        mQueueFamilyIndex);
}