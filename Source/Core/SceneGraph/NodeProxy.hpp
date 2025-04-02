#pragma once

#include "Core/Model/GPUGeometryData.h"
#include "Core/Vulkan/Manager/RenderFrame.h"
#include "Core/Vulkan/Native/DGCSequence.h"
#include "Node.h"

namespace IntelliDesign_NS::Vulkan::Core {
class RenderFrame;
}

namespace IntelliDesign_NS::Core::SceneGraph {

using Type_pSeqDataBufPool = Vulkan::Core::DGCSeqBase::SequenceDataBufferPool*;

template <class TDGCSeqTemp>
class NodeProxy : public Node {
public:
    NodeProxy(Node&& node, Type_pSeqDataBufPool pool)
        : Node(::std::move(node)), mPool(pool) {}

    ~NodeProxy() override;

    void RequestSeqBufIDs() override;

    void RetrieveIDs() override;

    ModelData::CISDI_3DModel const& SetModel(const char* modelPath) override;

    void UploadSeqBuf(Vulkan::Core::RenderFrame& frame) override;

    MemoryPool::Type_STLVector<uint64_t> const& GetIDs() const override;

private:
    MemoryPool::Type_STLVector<uint64_t> mSeqBufIDs {};
    Type_pSeqDataBufPool mPool {nullptr};
    int dirty = 3;
};

template <class TDGCSeqTemp>
NodeProxy<TDGCSeqTemp>::~NodeProxy() = default;

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::RequestSeqBufIDs() {
    mSeqBufIDs.reserve(mSequenceCount);
    for (uint32_t i = 0; i < mSequenceCount; ++i) {
        mSeqBufIDs.emplace_back(mPool->RequestID());
    }
    int dirty = 3;
}

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::RetrieveIDs() {
    for (auto id : mSeqBufIDs) {
        mPool->RetrieveID(id);
    }
    mSeqBufIDs.clear();
}

template <class TDGCSeqTemp>
ModelData::CISDI_3DModel const& NodeProxy<TDGCSeqTemp>::SetModel(
    const char* modelPath) {
    auto const& cisdiModel = Node::SetModel(modelPath);

    return cisdiModel;
}

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::UploadSeqBuf(Vulkan::Core::RenderFrame& frame) {
    // if (!dirty)
    //     return;

    for (uint32_t i = 0; i < mSequenceCount; ++i) {
        auto resHandle = mPool->GetResource(mSeqBufIDs[i]);

        // fill cpu data
        VE_ASSERT(resHandle, "pool resource invalid");

        VE_ASSERT(resHandle->size == sizeof(TDGCSeqTemp), "size dont match");

        auto pRes = static_cast<TDGCSeqTemp*>(resHandle->ptr);

        if constexpr (TDGCSeqTemp::_UseExecutionSet_) {
            pRes->index = 0;
        }

        if constexpr (!::std::is_same_v<
                          typename TDGCSeqTemp::_Type_PushConstant_, void>) {
            auto pc = mGPUGeoData->GetMeshletPushContants(i);

            if constexpr (!::std::is_same_v<
                              typename TDGCSeqTemp::_Type_PushConstant_,
                              decltype(pc)>) {
                throw ::std::runtime_error("push constant type dont match.");
            }

            pRes->pushConstant = pc;
            pRes->command = mGPUGeoData->GetDrawIndirectCmdBufInfo(i);
        }

        // copy cpu data to staging buffer
        void* data = resHandle->GetStagingMappedPtr(frame.GetIndex());
        memcpy(static_cast<char*>(data) + resHandle->id * sizeof(TDGCSeqTemp),
               pRes, sizeof(TDGCSeqTemp));

        // collect staging buffer & gpu buffer pair and copy size
        auto copyInfo = resHandle->GetCopyInfo(frame.GetIndex());

        auto key = ::std::pair<const char*, const char*> {copyInfo.srcName,
                                                          copyInfo.dstName};

        if (frame.mCmdStagings.contains(key)) {
            frame.mCmdStagings.at(key) += resHandle->size;
        } else {
            frame.mCmdStagings.emplace(key, resHandle->size);
        }
    }

    // dirty--;
}

template <class TDGCSeqTemp>
MemoryPool::Type_STLVector<uint64_t> const& NodeProxy<TDGCSeqTemp>::GetIDs()
    const {
    return mSeqBufIDs;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph