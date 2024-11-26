#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class CommandPool;

class CommandBuffer {
public:
    CommandBuffer(
        Context* ctx, CommandPool* pool,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    ~CommandBuffer() = default;
    CLASS_MOVABLE_ONLY(CommandBuffer);

public:
    vk::CommandBuffer GetHandle() const { return mCmdBuffer; }

    void Reset();
    void End();

private:
    vk::CommandBuffer CreateCommandBuffer();

private:
    Context* pContex;
    CommandPool* pCmdPool;
    vk::CommandBufferLevel mLevel;

    vk::CommandBuffer mCmdBuffer;
};

class CommandPool {
public:
    CommandPool(Context* ctx, uint32_t queueFamilysIndex,
                vk::CommandPoolCreateFlags flags =
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    ~CommandPool();
    CLASS_MOVABLE_ONLY(CommandPool);

public:
    vk::CommandPool GetHandle() const { return mCmdPool; }

    CommandBuffer& RequestCommandBuffer();
    void Reset();

private:
    vk::CommandPool CreateCommandPool();

private:
    Context* pCtx;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t mQueueFamilysIndex;

    vk::CommandPool mCmdPool;

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