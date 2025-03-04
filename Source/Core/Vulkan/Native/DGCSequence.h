#pragma once

#include "Buffer.h"
#include "Core/Utilities/Defines.h"
#include "DGCSequenceLayout.h"

#include "Core/Vulkan/Manager/PipelineManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

template <class TDGCSequenceTemplate>
class DGCSequence {
public:
    using Type_SequenceTemplate = TDGCSequenceTemplate;

    DGCSequence(VulkanContext& context, PipelineManager& pipelineMgr,
                uint32_t sequenceCount, uint32_t maxDrawCount,
                uint32_t maxPipelineCount);

    ~DGCSequence();

    DGCSequence& AddESObject(const char* name);

    void MakeSequenceLayout(const char* pipelineLayoutName,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    SequenceLayout& GetLayout() const;

    void MakeExecutionSet(const char* initialESObjectName);

    void MakeExecutionSet(vk::ShaderEXT initialShader);

    vk::IndirectExecutionSetEXT GetExecutionSet() const;

    Type_STLVector<TDGCSequenceTemplate>& GetSequenceData();

    vk::DeviceAddress GetSequenceDataBufferAddress() const;

    void Finalize();

private:
    void GenerateSequenceDataBuffer();

private:
    VulkanContext& mContext;
    PipelineManager& mPipelineMgr;

    uint32_t mMaxPipelineCount;
    uint32_t mSequenceCount;
    uint32_t mMaxDrawCount;

    Type_STLUnorderedMap_String<uint32_t> mObjectNamesIdxMap;

    vk::IndirectExecutionSetEXT mExecutionSet;

    Type_UniquePtr<SequenceLayout> mLayout;

    Type_STLVector<TDGCSequenceTemplate> mSequenceData;
    Type_SharedPtr<Buffer> mSequenceDataBuffer;
};

template <class TDGCSequenceTemplate>
DGCSequence<TDGCSequenceTemplate>::DGCSequence(VulkanContext& context,
                                               PipelineManager& pipelineMgr,
                                               uint32_t sequenceCount,
                                               uint32_t maxDrawCount,
                                               uint32_t maxPipelineCount)
    : mContext(context),
      mPipelineMgr(pipelineMgr),
      mMaxPipelineCount(maxPipelineCount),
      mSequenceCount(sequenceCount),
      mMaxDrawCount(maxDrawCount) {
    mObjectNamesIdxMap.reserve(maxPipelineCount);
}

template <class TDGCSequenceTemplate>
DGCSequence<TDGCSequenceTemplate>::~DGCSequence() {
    if (mExecutionSet)
        mContext.GetDevice()->destroy(mExecutionSet);
}

template <class TDGCSequenceTemplate>
DGCSequence<TDGCSequenceTemplate>&
DGCSequence<TDGCSequenceTemplate>::AddESObject(const char* name) {
    uint32_t idx = mObjectNamesIdxMap.size();
    mObjectNamesIdxMap.emplace(name, idx);
    return *this;
}

template <class TDGCSequenceTemplate>
void DGCSequence<TDGCSequenceTemplate>::MakeSequenceLayout(
    const char* pipelineLayoutName, bool unorderedSequence,
    bool explicitPreprocess) {
    auto pipelineLayout = mPipelineMgr.GetLayoutHandle(pipelineLayoutName);

    mLayout = CreateLayout<TDGCSequenceTemplate>(
        mContext, pipelineLayout, unorderedSequence, explicitPreprocess);
}

template <class TDGCSequenceTemplate>
SequenceLayout& DGCSequence<TDGCSequenceTemplate>::GetLayout() const {
    return *mLayout;
}

template <class TDGCSequenceTemplate>
void DGCSequence<TDGCSequenceTemplate>::MakeExecutionSet(
    const char* initialESObjectName) {
    static_assert(TDGCSequenceTemplate::_UseExecutionSet_,
                  "This sequence does not use execution set.");

    if (mObjectNamesIdxMap.contains(initialESObjectName)) {
        if (mObjectNamesIdxMap.at(initialESObjectName) != 0) {
            auto it = ::std::find_if(
                mObjectNamesIdxMap.begin(), mObjectNamesIdxMap.end(),
                [&](const auto& pair) { return pair.second == 0; });

            ::std::swap(it->second, mObjectNamesIdxMap.at(initialESObjectName));
        }
    } else {
        for (auto& [name, idx] : mObjectNamesIdxMap) {
            idx += 1;
        }
        mObjectNamesIdxMap.emplace(initialESObjectName, 0);
    }

    vk::IndirectExecutionSetPipelineInfoEXT esPipelineInfo {};
    esPipelineInfo
        .setInitialPipeline(mPipelineMgr.GetPipelineHandle(initialESObjectName))
        .setMaxPipelineCount(mMaxPipelineCount);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::ePipelines)
        .setInfo(&esPipelineInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetPipelineEXT> writeIES {};
    writeIES.reserve(mObjectNamesIdxMap.size());

    for (const auto& [name, idx] : mObjectNamesIdxMap) {
        vk::WriteIndirectExecutionSetPipelineEXT write {};
        write.setPipeline(mPipelineMgr.GetPipelineHandle(name.c_str()))
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetPipelineEXT(mExecutionSet,
                                                                writeIES);
}

template <class TDGCSequenceTemplate>
void DGCSequence<TDGCSequenceTemplate>::MakeExecutionSet(
    vk::ShaderEXT initialShader) {
    static_assert(TDGCSequenceTemplate::_UseExecutionSet_,
                  "This sequence does not use execution set.");

    vk::IndirectExecutionSetShaderInfoEXT esPipelineInfo {};
    // esPipelineInfo
    //     .setInitialPipeline(mPipelineMgr.GetPipelineHandle(initialESObjectName))
    //     .setMaxPipelineCount(mMaxPipelineCount);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::ePipelines)
        .setInfo(&esPipelineInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetPipelineEXT> writeIES {};
    writeIES.reserve(mObjectNamesIdxMap.size());

    for (const auto& [name, idx] : mObjectNamesIdxMap) {
        vk::WriteIndirectExecutionSetPipelineEXT write {};
        write.setPipeline(mPipelineMgr.GetPipelineHandle(name.c_str()))
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetPipelineEXT(mExecutionSet,
                                                                writeIES);
}

template <class TDGCSequenceTemplate>
vk::IndirectExecutionSetEXT DGCSequence<TDGCSequenceTemplate>::GetExecutionSet()
    const {
    VE_ASSERT(mExecutionSet != VK_NULL_HANDLE, "Execution Set is not created.");
    return mExecutionSet;
}

template <class TDGCSequenceTemplate>
Type_STLVector<TDGCSequenceTemplate>&
DGCSequence<TDGCSequenceTemplate>::GetSequenceData() {
    // VE_ASSERT((TDGCSequenceTemplate::_UseExecutionSet_
    //            && mExecutionSet != VK_NULL_HANDLE)
    //               || !TDGCSequenceTemplate::_UseExecutionSet_,
    //           "Sequence Data should be set after MakeExecutionSet");

    if (mSequenceData.empty())
        mSequenceData.resize(mSequenceCount);
    return mSequenceData;
}

template <class TDGCSequenceTemplate>
vk::DeviceAddress
DGCSequence<TDGCSequenceTemplate>::GetSequenceDataBufferAddress() const {
    VE_ASSERT(mSequenceDataBuffer, "Sequence Data Buffer is not created.");
    return mSequenceDataBuffer->GetDeviceAddress();
}

template <class TDGCSequenceTemplate>
void DGCSequence<TDGCSequenceTemplate>::Finalize() {
    GenerateSequenceDataBuffer();
}

template <class TDGCSequenceTemplate>
void DGCSequence<TDGCSequenceTemplate>::GenerateSequenceDataBuffer() {
    VE_ASSERT(!mSequenceData.empty(), "No sequence data was set.")

    auto const sequenceDataSize = sizeof(TDGCSequenceTemplate) * mSequenceCount;

    mSequenceDataBuffer = mContext.CreateDeviceLocalBuffer(
        "DGC sequence data", sequenceDataSize,
        vk::BufferUsageFlagBits::eIndirectBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst);

    auto staging = mContext.CreateStagingBuffer("", sequenceDataSize);

    memcpy(staging->GetMapPtr(), mSequenceData.data(), sequenceDataSize);

    {
        auto cmd = mContext.CreateCmdBufToBegin(
            mContext.GetQueue(QueueType::Transfer));
        vk::BufferCopy cmdBufCopy {};
        cmdBufCopy.setSize(sequenceDataSize);
        cmd->copyBuffer(staging->GetHandle(), mSequenceDataBuffer->GetHandle(),
                        cmdBufCopy);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core
