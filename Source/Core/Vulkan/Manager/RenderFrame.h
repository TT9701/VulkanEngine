#pragma once

#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/Descriptors.h"
#include "Core/Vulkan/Native/SyncStructures.h"
#include "Core/Vulkan/Native/QueryPool.h"

#include "Core/System/concurrentqueue.h"

namespace IntelliDesign_NS::Vulkan::Core {

class GPUGeometryData;

class RenderFrame {
    using Type_Task = ::std::function<void()>;
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

    Semaphore const& GetPresentFinishedSemaphore() const;
    Semaphore const& GetRenderFinishedSemaphore() const;
    Semaphore const& GetSwapchainPresentSemaphore() const;

    CmdBufferToBegin GetGraphicsCmdBuf() const;
    CmdBufferToBegin GetComputeCmdBuf() const;
    CmdBufferToBegin GetTransferCmdBuf() const;

    BindlessDescPool& GetBindlessDescPool() const;

    QueryPool& GetQueryPool() const;

    void Reset();

    void CullRegister(SharedPtr<GPUGeometryData> const& refData);

    void ClearGPUGeoDataRefs();

    Type_STLVector<const char*> mCmdStagings {};

private:
    VulkanContext& mContext;

    Type_STLMap<uint32_t, UniquePtr<CommandPool>> mCmdPools;

    UniquePtr<FencePool> mFencePool;
    UniquePtr<SemaphorePool> mSemaphorePool;

    UniquePtr<Semaphore> mPresentFinished;
    UniquePtr<Semaphore> mRenderFinished;

    UniquePtr<Semaphore> mSwapchainPresent;

    UniquePtr<QueryPool> mQueryPool;

    SharedPtr<BindlessDescPool> mBindlessDescPool;

    Type_STLVector<SharedPtr<GPUGeometryData>> mRefGPUGeoDatas {};

};
}  // namespace IntelliDesign_NS::Vulkan::Core