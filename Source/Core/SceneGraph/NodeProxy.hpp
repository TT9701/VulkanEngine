#pragma once

#include "Core/Model/GPUGeometryData.h"
#include "Core/Vulkan/Native/DGCSequence.h"
#include "Node.h"

namespace IntelliDesign_NS::Core::SceneGraph {

using Type_pSeqDataBufPool = Vulkan::Core::DGCSeqBase::SequenceDataBufferPool*;

template <class TDGCSeqTemp>
class NodeProxy : public Node {
    struct SeqBufIDInfo {
        uint64_t id;
        bool dirty {true};
    };

public:
    NodeProxy(Node&& node, Type_pSeqDataBufPool pool)
        : Node(::std::move(node)), mPool(pool) {}

    ~NodeProxy() override;

    void RequestSeqBufIDs() override;

    void RetrieveIDs() override;

    ModelData::CISDI_3DModel const& SetModel(const char* modelPath) override;

    void UploadSeqBuf() override;

    MemoryPool::Type_STLVector<MemoryPool::Type_STLVector<Type_CopyInfo>> const&
    GetCopyInfos() const;

private:
    MemoryPool::Type_STLVector<SeqBufIDInfo> mSeqBufIDs {};
    Type_pSeqDataBufPool mPool {nullptr};
    MemoryPool::Type_STLVector<MemoryPool::Type_STLVector<Type_CopyInfo>>
        mCopyInfos {};
};

template <class TDGCSeqTemp>
NodeProxy<TDGCSeqTemp>::~NodeProxy() {
    RetrieveIDs();
}

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::RequestSeqBufIDs() {
    uint32_t currentIDCount = mSeqBufIDs.size();

    if (currentIDCount < mSequenceCount) {
        mSeqBufIDs.reserve(mSequenceCount);
        for (uint32_t i = currentIDCount; i < mSequenceCount; ++i) {
            mSeqBufIDs.emplace_back(mPool->RequestID());
        }
    }
}

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::RetrieveIDs() {
    for (auto idInfo : mSeqBufIDs) {
        mPool->RetrieveID(idInfo.id);
    }
    mSeqBufIDs.clear();
}

template <class TDGCSeqTemp>
ModelData::CISDI_3DModel const& NodeProxy<TDGCSeqTemp>::SetModel(
    const char* modelPath) {
    auto const& cisdiModel = Node::SetModel(modelPath);

    RequestSeqBufIDs();

    return cisdiModel;
}

template <class TDGCSeqTemp>
void NodeProxy<TDGCSeqTemp>::UploadSeqBuf() {
    for (uint32_t i = 0; i < mSequenceCount; ++i) {
        auto& idInfo = mSeqBufIDs[i];

        if (idInfo.dirty) {
            auto resHandle = mPool->GetResource(idInfo.id);

            VE_ASSERT(resHandle, "pool resource invalid");

            VE_ASSERT(resHandle->size == sizeof(TDGCSeqTemp),
                      "size dont match");

            auto pRes = reinterpret_cast<TDGCSeqTemp*>(resHandle->ptr);

            if constexpr (TDGCSeqTemp::_UseExecutionSet_) {
                pRes->index = 0;
            }

            if constexpr (!::std::is_same_v<
                              typename TDGCSeqTemp::_Type_PushConstant_,
                              void>) {
                auto pc = mGPUGeoData->GetMeshletPushContants(i);

                if constexpr (!::std::is_same_v<
                                  typename TDGCSeqTemp::_Type_PushConstant_,
                                  decltype(pc)>) {
                    throw ::std::runtime_error(
                        "push constant type dont match.");
                }

                pRes->pushConstant = pc;
                pRes->command = mGPUGeoData->GetDrawIndirectCmdBufInfo(i);
            }
            idInfo.dirty = false;

            auto& vecInfos = mCopyInfos.emplace_back();
            for (uint32_t i = 0; i < 3; ++i) {
                vecInfos.emplace_back(resHandle->GetCopyInfo(i));
            }
        }
    }
}

template <class TDGCSeqTemp>
MemoryPool::Type_STLVector<MemoryPool::Type_STLVector<Type_CopyInfo>> const&
NodeProxy<TDGCSeqTemp>::GetCopyInfos() const {
    return mCopyInfos;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph