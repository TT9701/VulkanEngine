#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class CommandPool {
public:
    CommandPool(Context* ctx, uint32_t queueFamilysIndex,
                vk::CommandPoolCreateFlags flags =
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    ~CommandPool();
    CLASS_MOVABLE_ONLY(CommandPool);

public:
    vk::CommandPool GetHandle() const { return mCmdPool; }

private:
    vk::CommandPool CreateCommandPool();

private:
    Context* pCtx;
    vk::CommandPoolCreateFlags mFlags;
    uint32_t mQueueFamilysIndex;

    vk::CommandPool mCmdPool;
};

class CommandBuffers {
public:
    CommandBuffers(
        Context* ctx, CommandPool* pool, uint32_t count = 1u,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    ~CommandBuffers() = default;
    CLASS_MOVABLE_ONLY(CommandBuffers);

public:
    vk::CommandBuffer GetHandle(uint32_t index = 0) const {
        return mCmdBuffer[index];
    }

    uint32_t GetBufferCount() const {
        return static_cast<uint32_t>(mCmdBuffer.size());
    }

private:
    Type_STLVector<vk::CommandBuffer> CreateCommandBuffers(uint32_t count);

private:
    Context* pContex;
    CommandPool* pCmdPool;
    vk::CommandBufferLevel mLevel;

    Type_STLVector<vk::CommandBuffer> mCmdBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core