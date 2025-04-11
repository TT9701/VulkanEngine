#pragma once

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Commands.h"
#include "Core/Vulkan/Native/Descriptors.h"
#include "Core/Vulkan/Native/QueryPool.h"
#include "Core/Vulkan/Native/RenderResource.h"
#include "Core/Vulkan/Native/SyncStructures.h"

namespace IntelliDesign_NS::Core::SceneGraph {
class Node;
}

namespace IntelliDesign_NS::Vulkan::Core {

class GPUGeometryData;
class RenderResourceManager;

class RenderFrame {
public:
    RenderFrame(VulkanContext& context, RenderResourceManager& renderResMgr,
                uint32_t idx);

    void PrepareBindlessDescPool(
        Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
        vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler);

    uint32_t GetIndex() const;

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

    void CullRegister(
        SharedPtr<IntelliDesign_NS::Core::SceneGraph::Node> const& node);

    void ClearNodes();

    Type_STLVector<SharedPtr<IntelliDesign_NS::Core::SceneGraph::Node>>&
    GetInFrustumNodes();

    const char* GetReadbackBufferName() const;

    RenderResource const& GetReadbackBuffer() const;

    Type_STLMap<::std::pair<const char*, const char*>, size_t> mCmdStagings {};

private:
    VulkanContext& mContext;
    RenderResourceManager& mRenderResMgr;

    uint32_t mIdx;

    Type_STLMap<uint32_t, UniquePtr<CommandPool>> mCmdPools;

    UniquePtr<FencePool> mFencePool;
    UniquePtr<SemaphorePool> mSemaphorePool;

    UniquePtr<Semaphore> mPresentFinished;
    UniquePtr<Semaphore> mRenderFinished;

    UniquePtr<Semaphore> mSwapchainPresent;

    UniquePtr<QueryPool> mQueryPool;

    Type_STLString mReadbackBufferName;

    SharedPtr<BindlessDescPool> mBindlessDescPool;

    Type_STLVector<SharedPtr<IntelliDesign_NS::Core::SceneGraph::Node>>
        mNodes {};
};
}  // namespace IntelliDesign_NS::Vulkan::Core