#include "CommandBuffer.h"

#include "CommandPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

CommandBuffer::CommandBuffer(HPPCommandPool& commandPool,
                             vk::CommandBufferLevel level)
    : VulkanResource(nullptr, &commandPool.GetDevice()),
      mLevel(level),
      mCommandPool(commandPool) {
    vk::CommandBufferAllocateInfo info(commandPool.GetHandle(), level, 1);
    SetHandle(GetDevice().GetHandle().allocateCommandBuffers(info).front());
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : VulkanResource(std::move(other)),
      mLevel(other.mLevel),
      mCommandPool(other.mCommandPool),
      mStoredPushConstants(std::exchange(other.mStoredPushConstants, {})),
      mMaxPushConstantsSize(std::exchange(other.mMaxPushConstantsSize, {})),
      mLastFramebufferExtent(std::exchange(other.mLastFramebufferExtent, {})),
      mLastRenderAreaExtent(std::exchange(other.mLastRenderAreaExtent, {})) {}

CommandBuffer::~CommandBuffer() {
    if (GetHandle() != VK_NULL_HANDLE) {
        GetDevice().GetHandle().freeCommandBuffers(mCommandPool.GetHandle(),
                                                   GetHandle());
    }
}

vk::Result CommandBuffer::Reset(ResetMode reset_mode) {
    assert(reset_mode == mCommandPool.GetResetMode() && "Command buffer reset mode must match the one used by the pool to allocate it");

    if (reset_mode == ResetMode::ResetIndividually) {
        GetHandle().reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    }

    return vk::Result::eSuccess;
}

}  // namespace IntelliDesign_NS::Vulkan::Core