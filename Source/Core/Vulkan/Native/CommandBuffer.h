#pragma once

#include "VulkanResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class HPPCommandPool;

class CommandBuffer : public VulkanResource<vk::CommandBuffer> {
public:
    enum class ResetMode {
        ResetPool,
        ResetIndividually,
        AlwaysAllocate,
    };

public:
    CommandBuffer(HPPCommandPool& commandPool, vk::CommandBufferLevel level);
    CommandBuffer(CommandBuffer&& other) noexcept;
    ~CommandBuffer() override;

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer& operator=(CommandBuffer&&) = delete;

    vk::Result Begin(vk::CommandBufferUsageFlags flags,
                     CommandBuffer* primaryCmdBuf = nullptr);

	vk::Result Reset(ResetMode reset_mode);

private:
    vk::CommandBufferLevel const mLevel = {};
    HPPCommandPool& mCommandPool;
    std::vector<uint8_t> mStoredPushConstants = {};
    uint32_t mMaxPushConstantsSize = {};
    vk::Extent2D mLastFramebufferExtent = {};
    vk::Extent2D mLastRenderAreaExtent = {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core