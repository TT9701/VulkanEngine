#pragma once

#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/Descriptors.h"
#include "Core/Vulkan/Native/SyncStructures.h"

namespace IntelliDesign_NS::Vulkan::Core {
class RenderFrame {
public:
    RenderFrame(VulkanContext* context);

    void PrepareBindlessDescPool(
        Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
        vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler);

    FencePool& GetFencePool() const;
    Semaphore const& GetReady4RenderSemaphore() const;
    Semaphore const& GetReady4PresentSemaphore() const;

    CmdBufferToBegin GetGfxCmdBuf() const;
    CmdBufferToBegin GetCmpCmdBuf() const;
    CmdBufferToBegin GetTsfCmdBuf() const;

    BindlessDescPool& GetBindlessDescPool() const;

    void Reset();

private:
    VulkanContext* pContext;

    Type_STLMap<uint32_t, UniquePtr<CommandPool>> mCmdPools;

    UniquePtr<FencePool> mFencePool;
    UniquePtr<Semaphore> mReady4Render;
    UniquePtr<Semaphore> mReady4Present;

    SharedPtr<BindlessDescPool> mBindlessDescPool;
};
}  // namespace IntelliDesign_NS::Vulkan::Core