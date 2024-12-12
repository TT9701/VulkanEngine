#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class CommandPool;

class CommandBuffer {
public:
    CommandBuffer(
        VulkanContext& ctx, CommandPool& pool,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    ~CommandBuffer() = default;
    CLASS_MOVABLE_ONLY(CommandBuffer);

public:
    vk::CommandBuffer GetHandle() const;

    vk::CommandBuffer const* operator->() const;

    void Reset();
    void End();

private:
    vk::CommandBuffer CreateCommandBuffer();

private:
    VulkanContext& mContex;
    CommandPool& mCmdPool;
    vk::CommandBufferLevel mLevel;

    vk::CommandBuffer mHandle;
};

class CommandPool {
public:
    CommandPool(VulkanContext& ctx, uint32_t queueFamilysIndex,
                vk::CommandPoolCreateFlags flags =
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    ~CommandPool();
    CLASS_MOVABLE_ONLY(CommandPool);

public:
    vk::CommandPool GetHandle() const;

    CommandBuffer& RequestCommandBuffer(
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);

    void Reset();

private:
    vk::CommandPool CreateCommandPool();

private:
    VulkanContext& mContext;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t mQueueFamilysIndex;

    vk::CommandPool mHandle;

    uint32_t mActiveCmdBufCount {0};
    Type_STLVector<UniquePtr<CommandBuffer>> mCmdBuffers;
};

struct CmdBufferToBegin {
    CmdBufferToBegin(CommandBuffer& cmd);

    vk::CommandBuffer GetHandle() const;

    void End();

private:
    CommandBuffer& mBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core