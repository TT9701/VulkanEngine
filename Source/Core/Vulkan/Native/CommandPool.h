#pragma once

#include "CommandBuffer.h"
#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderFrame;
class Device;

class HPPCommandPool : public VulkanResource<vk::CommandPool> {
public:
    HPPCommandPool(Device& device, uint32_t queueFamilyIndex,
                   RenderFrame* renderFrame = nullptr, size_t threadIndex = 0,
                   CommandBuffer::ResetMode resetMode =
                       CommandBuffer::ResetMode::ResetPool);
    HPPCommandPool(HPPCommandPool&& other) noexcept;
    ~HPPCommandPool() override;

    HPPCommandPool(const HPPCommandPool&) = delete;
    HPPCommandPool& operator=(const HPPCommandPool&) = delete;
    HPPCommandPool& operator=(HPPCommandPool&&) = delete;

    uint32_t GetQueueFamilyIndex() const;
    RenderFrame* GetRenderFrame();
    CommandBuffer::ResetMode GetResetMode() const;
    size_t GetThreadIndex() const;
    CommandBuffer& RequestCommandBuffer(
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    void ResetPool();

private:
    void ResetCommandBuffers();

private:
    RenderFrame* mRenderFrame = nullptr;
    size_t mThreadIndex = 0;
    uint32_t mQueueFamilyIndex = 0;
    std::vector<UniquePtr<CommandBuffer>> mPrimaryCommandBuffers;
    uint32_t mActivePrimaryCommandBufferCount = 0;
    std::vector<UniquePtr<CommandBuffer>> mSecondaryCommandBuffers;
    uint32_t mActiveSecondaryCommandBufferCount = 0;
    CommandBuffer::ResetMode mResetMode = CommandBuffer::ResetMode::ResetPool;
};

}  // namespace IntelliDesign_NS::Vulkan::Core