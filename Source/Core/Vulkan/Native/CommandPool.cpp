#include "CommandPool.h"

namespace IntelliDesign_NS::Vulkan::Core {
HPPCommandPool::HPPCommandPool(Device& device, uint32_t queueFamilyIndex,
                               RenderFrame* renderFrame, size_t threadIndex,
                               CommandBuffer::ResetMode resetMode)
    : VulkanResource(VK_NULL_HANDLE, &device),
      mRenderFrame {renderFrame},
      mThreadIndex {threadIndex},
      mResetMode {resetMode} {
    vk::CommandPoolCreateFlags flags;
    switch (resetMode) {
        case CommandBuffer::ResetMode::ResetIndividually:
        case CommandBuffer::ResetMode::AlwaysAllocate:
            flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
            break;
        case CommandBuffer::ResetMode::ResetPool:
        default: flags = vk::CommandPoolCreateFlagBits::eTransient; break;
    }

    vk::CommandPoolCreateInfo command_pool_create_info(flags, queueFamilyIndex);

    GetHandle() =
        device.GetHandle().createCommandPool(command_pool_create_info);
}

HPPCommandPool::HPPCommandPool(HPPCommandPool&& other) noexcept
    : VulkanResource(::std::move(other)),
      mRenderFrame(std::exchange(other.mRenderFrame, {})),
      mThreadIndex(std::exchange(other.mThreadIndex, {})),
      mQueueFamilyIndex(std::exchange(other.mQueueFamilyIndex, {})),
      mPrimaryCommandBuffers {std::move(other.mPrimaryCommandBuffers)},
      mActivePrimaryCommandBufferCount(
          std::exchange(other.mActivePrimaryCommandBufferCount, {})),
      mSecondaryCommandBuffers {std::move(other.mSecondaryCommandBuffers)},
      mActiveSecondaryCommandBufferCount(
          std::exchange(other.mActiveSecondaryCommandBufferCount, {})),
      mResetMode(std::exchange(other.mResetMode, {})) {}

HPPCommandPool::~HPPCommandPool() {
    mPrimaryCommandBuffers.clear();
    mSecondaryCommandBuffers.clear();

    if (auto& handle = GetHandle()) {
        GetDevice().GetHandle().destroyCommandPool(handle);
    }
}

uint32_t HPPCommandPool::GetQueueFamilyIndex() const {
    return mQueueFamilyIndex;
}

RenderFrame* HPPCommandPool::GetRenderFrame() {
    return mRenderFrame;
}

CommandBuffer::ResetMode HPPCommandPool::GetResetMode() const {
    return mResetMode;
}

size_t HPPCommandPool::GetThreadIndex() const {
    return mThreadIndex;
}

CommandBuffer& HPPCommandPool::RequestCommandBuffer(
    vk::CommandBufferLevel level) {
    if (level == vk::CommandBufferLevel::ePrimary) {
        if (mActivePrimaryCommandBufferCount < mPrimaryCommandBuffers.size()) {
            return *mPrimaryCommandBuffers[mActivePrimaryCommandBufferCount++];
        }

        mPrimaryCommandBuffers.emplace_back(
            MakeUnique<CommandBuffer>(*this, level));

        mActivePrimaryCommandBufferCount++;

        return *mPrimaryCommandBuffers.back();
    } else {
        if (mActiveSecondaryCommandBufferCount
            < mSecondaryCommandBuffers.size()) {
            return *mSecondaryCommandBuffers
                [mActiveSecondaryCommandBufferCount++];
        }

        mSecondaryCommandBuffers.emplace_back(
            MakeUnique<CommandBuffer>(*this, level));

        mActiveSecondaryCommandBufferCount++;

        return *mSecondaryCommandBuffers.back();
    }
}

void HPPCommandPool::ResetPool() {
    auto& handle = GetHandle();
    switch (mResetMode) {
        case CommandBuffer::ResetMode::ResetIndividually:
            ResetCommandBuffers();
            break;

        case CommandBuffer::ResetMode::ResetPool:
            GetDevice().GetHandle().resetCommandPool(handle);
            ResetCommandBuffers();
            break;

        case CommandBuffer::ResetMode::AlwaysAllocate:
            mPrimaryCommandBuffers.clear();
            mActivePrimaryCommandBufferCount = 0;
            mSecondaryCommandBuffers.clear();
            mActiveSecondaryCommandBufferCount = 0;
            break;

        default:
            throw std::runtime_error("Unknown reset mode for command pools");
    }
}

void HPPCommandPool::ResetCommandBuffers() {
    for (auto& cmd_buf : mPrimaryCommandBuffers) {
        cmd_buf->Reset(mResetMode);
    }
    mActivePrimaryCommandBufferCount = 0;

    for (auto& cmd_buf : mSecondaryCommandBuffers) {
        cmd_buf->Reset(mResetMode);
    }
    mActiveSecondaryCommandBufferCount = 0;
}

}  // namespace IntelliDesign_NS::Vulkan::Core