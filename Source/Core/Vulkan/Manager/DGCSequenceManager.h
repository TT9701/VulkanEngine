#pragma once

#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Native/DGCSequence.h"

namespace IntelliDesign_NS::Vulkan::Core {

class SequenceDataBuffer {
public:
    SequenceDataBuffer(VulkanContext& context, RenderResource& buffer,
                       uint32_t seqCount, uint32_t stride);

public:
    template <class T>
    class Data {
    public:
        explicit Data(UniquePtr<SequenceDataBuffer>&& buffer)
            : data(buffer->GetDataPtr<T>()), buffer(::std::move(buffer)) {}

        ~Data();

        T* data {};

    private:
        UniquePtr<SequenceDataBuffer> buffer;
    };

public:
    template <class T>
    static Data<T> GetData(UniquePtr<SequenceDataBuffer>&& buffer) {
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

struct DGCSequenceInfo_Pipeline {
    uint32_t maxSequenceCount {0};
    uint32_t maxDrawCount {0};
    uint32_t maxPipelineCount {0};

    const char* initialPipelineName = nullptr;

    bool unorderedSequence = false;
    bool explicitPreprocess = false;

    const char* pipelineLayoutName = nullptr;
};

struct DGCSequenceInfo_Shader {
    uint32_t maxSequenceCount {0};
    uint32_t maxDrawCount {0};
    uint32_t maxShaderCount {0};

    Type_STLVector<ShaderIDInfo> initialShaderIdInfos {};

    bool unorderedSequence = false;
    bool explicitPreprocess = false;
};

class DGCSequenceManager {
public:
    DGCSequenceManager(VulkanContext& context, PipelineManager& pipelineMgr,
                       ShaderManager& shaderMgr,
                       RenderResourceManager& renderResMgr);

public:
    template <class TDGCSequenceTemplate>
    DGCSequence_ESPipeline& CreateSequence(
        DGCSequenceInfo_Pipeline const& info,
        ::std::initializer_list<const char*> pipelineNamesInES = {});

    template <class TDGCSequenceTemplate>
    DGCSequence_ESShader& CreateSequence(
        DGCSequenceInfo_Shader const& info,
        ::std::initializer_list<ShaderIDInfo> shaderInfosInES = {});

    template <class TDGCSequenceTemplate>
    SequenceDataBuffer::Data<TDGCSequenceTemplate> CreateDataBuffer(
        const char* name);

public:
    template <class TDGCSequenceTemplate>
    DGCSequenceBase& GetSequence();

    DGCSequenceBase& GetSequence(const char* name);

private:
    VulkanContext& mContext;
    PipelineManager& mPipelineMgr;
    ShaderManager& mShaderMgr;
    RenderResourceManager& mRenderResMgr;

    Type_STLUnorderedMap_String<SharedPtr<DGCSequenceBase>> mSequences {};
};

template <class T>
SequenceDataBuffer::Data<T>::~Data() {
    buffer->CreateSequenceDataBuffer();
}

template <class TDGCSequenceTemplate>
DGCSequence_ESPipeline& DGCSequenceManager::CreateSequence(
    DGCSequenceInfo_Pipeline const& info,
    std::initializer_list<const char*> pipelineNamesInES) {
    auto ptr = MakeShared<DGCSequence_ESPipeline>(
        mContext, mPipelineMgr, info.maxSequenceCount, info.maxDrawCount,
        info.maxPipelineCount);

    ptr->MakeSequenceLayout<TDGCSequenceTemplate>(
        info.pipelineLayoutName ? info.pipelineLayoutName
                                : info.initialPipelineName,
        info.unorderedSequence, info.explicitPreprocess);

    for (const auto& name : pipelineNamesInES) {
        ptr->AddPipeline(name);
    }

    ptr->MakeExecutionSet(info.initialPipelineName);
    ptr->Finalize();

    const auto pRes = ptr.get();

    auto const& [it, success] = mSequences.try_emplace(
        typeid(TDGCSequenceTemplate).name(), std::move(ptr));

    return *pRes;
}

template <class TDGCSequenceTemplate>
DGCSequence_ESShader& DGCSequenceManager::CreateSequence(
    DGCSequenceInfo_Shader const& info,
    std::initializer_list<ShaderIDInfo> shaderInfosInES) {
    auto ptr = MakeShared<DGCSequence_ESShader>(
        mContext, mShaderMgr, info.maxSequenceCount, info.maxDrawCount,
        info.maxShaderCount);

    ptr->MakeSequenceLayout<TDGCSequenceTemplate>(info.initialShaderIdInfos,
                                                  info.unorderedSequence,
                                                  info.explicitPreprocess);

    for (const auto& idInfo : shaderInfosInES) {
        ptr->AddShader(idInfo);
    }

    ptr->MakeExecutionSet(info.initialShaderIdInfos);
    ptr->Finalize();

    const auto pRes = ptr.get();

    auto const& [it, success] = mSequences.try_emplace(
        typeid(TDGCSequenceTemplate).name(), std::move(ptr));

    return *pRes;
}

template <class TDGCSequenceTemplate>
SequenceDataBuffer::Data<TDGCSequenceTemplate>
DGCSequenceManager::CreateDataBuffer(const char* name) {
    auto seqTemplateName = typeid(TDGCSequenceTemplate).name();
    auto sequencePtr = mSequences.at(seqTemplateName).get();

    auto stride = sizeof(TDGCSequenceTemplate);
    auto sequenceCount = sequencePtr->GetSequenceCount();

    auto& buf = mRenderResMgr.CreateBuffer(
        name, stride * sequenceCount,
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst,
        Buffer::MemoryType::DeviceLocal, stride);

    buf.SetBufferDGCSequence(
        mSequences.at(typeid(TDGCSequenceTemplate).name()));

    return SequenceDataBuffer::GetData<TDGCSequenceTemplate>(
        MakeUnique<SequenceDataBuffer>(mContext, buf, sequenceCount, stride));
}

template <class TDGCSequenceTemplate>
DGCSequenceBase& DGCSequenceManager::GetSequence() {
    return GetSequence(typeid(TDGCSequenceTemplate).name());
}

}  // namespace IntelliDesign_NS::Vulkan::Core