#pragma once

#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/Descriptors.h"
#include "Core/Vulkan/Native/SyncStructures.h"

namespace IntelliDesign_NS::Vulkan::Core {
class RenderFrame {
public:
    RenderFrame(VulkanContext& context);

    void PrepareBindlessDescPool(
        Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
        vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler);

    FencePool& GetFencePool() const;

    vk::Fence RequestFence(vk::FenceCreateFlags flags = {});

    SemaphorePool& GetSemaphorePool() const;

    vk::Semaphore RequestSemaphore();

    vk::Semaphore RequestSemaphore_WithOwnership();

    void ReleaseOwnedSemaphore(vk::Semaphore semaphore);


    Semaphore const& GetReady4RenderSemaphore() const;
    Semaphore const& GetReady4PresentSemaphore() const;

    CmdBufferToBegin GetGraphicsCmdBuf() const;
    CmdBufferToBegin GetComputeCmdBuf() const;
    CmdBufferToBegin GetTransferCmdBuf() const;

    BindlessDescPool& GetBindlessDescPool() const;

    void Reset();

private:
    VulkanContext& mContext;

    Type_STLMap<uint32_t, UniquePtr<CommandPool>> mCmdPools;

    UniquePtr<FencePool> mFencePool;
    UniquePtr<SemaphorePool> mSemaphorePool;

    UniquePtr<Semaphore> mReady4Render;
    UniquePtr<Semaphore> mReady4Present;

    SharedPtr<BindlessDescPool> mBindlessDescPool;
};
}  // namespace IntelliDesign_NS::Vulkan::Core