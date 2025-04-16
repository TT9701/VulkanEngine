#pragma once

#include <cstdint>
#include <initializer_list>
#include "Core/System/MemoryPool/MemoryPool.h"
#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Native/DGCSequence.h"

namespace IntelliDesign_NS::Vulkan::Core {

class DGCSeqDataBuffer {
public:
    DGCSeqDataBuffer(VulkanContext& context, RenderResource& buffer,
                     uint32_t seqCount, uint32_t stride);

public:
    template <class T>
    class Data {
    public:
        explicit Data(UniquePtr<DGCSeqDataBuffer>&& buffer)
            : data(buffer->GetDataPtr<T>()), buffer(::std::move(buffer)) {}

        ~Data();

        T* data {};

    private:
        UniquePtr<DGCSeqDataBuffer> buffer;
    };

public:
    template <class T>
    static Data<T> GetData(UniquePtr<DGCSeqDataBuffer>&& buffer) {
        return Data<T>(::std::move(buffer));
    }

private:
    template <class T>
    T* GetDataPtr() {
        return reinterpret_cast<T*>(sequenceData.data());
    }

    void CreateSequenceDataBuffer();

    VulkanContext& context;
    Type_STLVector<::std::byte> sequenceData {};
    RenderResource& buffer;
};

class DGCSeqManager {
    using Type_Sequences = Type_STLUnorderedMap_String<SharedPtr<DGCSeqBase>>;

public:
    DGCSeqManager(VulkanContext& context, PipelineManager& pipelineMgr,
                  ShaderManager& shaderMgr,
                  RenderResourceManager& renderResMgr);

public:
    template <class TDGCSeqTemplate>
    DGCSeq_ESPipeline& CreateSequence(DGCSeqInfo_Pipeline const& info);

    template <class TDGCSeqTemplate>
    DGCSeq_ESShader& CreateSequence(DGCSeqInfo_Shader const& info);

    template <class TDGCSeqTemplate>
    DGCSeqDataBuffer::Data<TDGCSeqTemplate> CreateDataBuffer(
        const char* name, DGCSeqBase const& seq);

private:
    template <class TDGCSeqTemplate>
    void CreateSequenceBufferPool(DGCSeqBase* seq);

public:
    template <class TDGCSeqTemplate>
    DGCSeqBase& GetSequence(DGCSeqInfo_Pipeline const& info);

    template <class TDGCSeqTemplate>
    DGCSeqBase& GetSequence(DGCSeqInfo_Shader const& info);

    DGCSeqBase& GetSequence(const char* name);

    SharedPtr<DGCSeqBase> GetSequenceRef(const char* name);

    Type_Sequences const& GetAllSequences() const;

private:
    VulkanContext& mContext;
    PipelineManager& mPipelineMgr;
    ShaderManager& mShaderMgr;
    RenderResourceManager& mRenderResMgr;

    Type_Sequences mSequences {};
};

template <class T>
DGCSeqDataBuffer::Data<T>::~Data() {
    buffer->CreateSequenceDataBuffer();
}

template <class TDGCSeqTemplate>
DGCSeq_ESPipeline& DGCSeqManager::CreateSequence(
    DGCSeqInfo_Pipeline const& info) {
    auto ptr = MakeShared<DGCSeq_ESPipeline>(
        mContext, mPipelineMgr, info.maxSequenceCount, info.maxDrawCount,
        info.maxPipelineCount);

    ptr->MakeSequenceLayout<TDGCSeqTemplate>(
        info.pipelineLayoutName ? info.pipelineLayoutName.c_str()
                                : info.initialPipelineName.c_str(),
        info.explicitPreprocess);

    for (const auto& name : info.pipelineNamesInES) {
        ptr->AddPipeline(name.c_str());
    }

    ptr->MakeExecutionSet(info.initialPipelineName.c_str());
    ptr->Finalize();

    const auto pRes = ptr.get();

    Type_STLString name = GetDGCSeqName<TDGCSeqTemplate>(info);

    ptr->SetName(name.c_str());

    auto const& [it, success] = mSequences.try_emplace(name, std::move(ptr));

    // create related buffer pool
    CreateSequenceBufferPool<TDGCSeqTemplate>(pRes);

    return *pRes;
}

template <class TDGCSeqTemplate>
DGCSeq_ESShader& DGCSeqManager::CreateSequence(DGCSeqInfo_Shader const& info) {
    auto ptr = MakeShared<DGCSeq_ESShader>(
        mContext, mPipelineMgr, mShaderMgr, info.maxSequenceCount,
        info.maxDrawCount, info.maxShaderCount);

    ptr->MakeSequenceLayout<TDGCSeqTemplate>(info);

    for (const auto& idInfo : info.shaderInfosInES) {
        ptr->AddShader(idInfo);
    }

    ptr->MakeExecutionSet(info.initialShaderIdInfos);
    ptr->Finalize();

    const auto pRes = ptr.get();

    Type_STLString name = GetDGCSeqName<TDGCSeqTemplate>(info);

    ptr->SetName(name.c_str());

    auto const& [it, success] = mSequences.try_emplace(name, std::move(ptr));

    // create related buffer pool
    CreateSequenceBufferPool<TDGCSeqTemplate>(pRes);

    return *pRes;
}

template <class TDGCSeqTemplate>
DGCSeqDataBuffer::Data<TDGCSeqTemplate> DGCSeqManager::CreateDataBuffer(
    const char* name, DGCSeqBase const& seq) {
    auto seqTemplateName = seq.GetName();
    auto sequencePtr = mSequences.at(seqTemplateName).get();

    auto stride = sizeof(TDGCSeqTemplate);
    auto sequenceCount = sequencePtr->GetSequenceCount();

    auto& buf = mRenderResMgr.CreateBuffer(
        name, stride * sequenceCount,
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst,
        Buffer::MemoryType::DeviceLocal, stride);

    buf.SetBufferDGCSequence(mSequences.at(seqTemplateName));

    return DGCSeqDataBuffer::GetData<TDGCSeqTemplate>(
        MakeUnique<DGCSeqDataBuffer>(mContext, buf, sequenceCount, stride));
}

template <class TDGCSeqTemplate>
void DGCSeqManager::CreateSequenceBufferPool(DGCSeqBase* seq) {
    seq->mSeqDataBufPool = MakeUnique<DGCSeqBase::SequenceDataBufferPool>(
        seq->GetSequenceCount(), mContext, mRenderResMgr, *this,
        seq->GetName(), sizeof(TDGCSeqTemplate));
}

template <class TDGCSeqTemplate>
DGCSeqBase& DGCSeqManager::GetSequence(DGCSeqInfo_Pipeline const& info) {
    auto name = GetDGCSeqName<TDGCSeqTemplate>(info);
    return GetSequence(name.c_str());
}

template <class TDGCSeqTemplate>
DGCSeqBase& DGCSeqManager::GetSequence(DGCSeqInfo_Shader const& info) {
    auto name = GetDGCSeqName<TDGCSeqTemplate>(info);
    return GetSequence(name.c_str());
}

}  // namespace IntelliDesign_NS::Vulkan::Core