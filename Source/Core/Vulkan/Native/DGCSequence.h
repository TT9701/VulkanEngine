#pragma once

#include "Buffer.h"
#include "Core/Utilities/Defines.h"
#include "DGCSequenceLayout.h"

#include "Core/Vulkan/Manager/PipelineManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

template <bool IsCompute, bool UsePipeline, class TPushConstant = void>
class DGCSequence {
public:
    using Type_SequenceLayout =
        DGCSequenceLayout<IsCompute, UsePipeline, TPushConstant>;

    using Type_SequenceTemplate =
        typename Type_SequenceLayout::Type_SequenceTemplate;

    DGCSequence(VulkanContext& context, PipelineManager& pipelineMgr,
                uint32_t sequenceCount, uint32_t maxDrawCount,
                uint32_t maxPipelineCount);

    ~DGCSequence();

    DGCSequence& AddPipeline(const char* name);

    void MakeSequenceLayout(const char* pipelineLayoutName,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    Type_SequenceLayout& GetLayout() const;

    void MakeExecutionSet(const char* initialPipelineName);

    vk::IndirectExecutionSetEXT GetExecutionSet() const;

    Type_STLVector<Type_SequenceTemplate>& GetSequenceData();

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

    Type_STLUnorderedMap_String<uint32_t> mPipelineNamesIdxMap;

    vk::IndirectExecutionSetEXT mExecutionSet;

    Type_UniquePtr<Type_SequenceLayout> mLayout;

    Type_STLVector<Type_SequenceTemplate> mSequenceData;
    Type_SharedPtr<Buffer> mSequenceDataBuffer;
};

template <bool IsCompute, bool UsePipeline, class TPushConstant>
DGCSequence<IsCompute, UsePipeline, TPushConstant>::DGCSequence(
    VulkanContext& context, PipelineManager& pipelineMgr,
    uint32_t sequenceCount, uint32_t maxDrawCount, uint32_t maxPipelineCount)
    : mContext(context),
      mPipelineMgr(pipelineMgr),
      mMaxPipelineCount(maxPipelineCount),
      mSequenceCount(sequenceCount),
      mMaxDrawCount(maxDrawCount) {
    mPipelineNamesIdxMap.reserve(maxPipelineCount);
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
DGCSequence<IsCompute, UsePipeline, TPushConstant>::~DGCSequence() {
    if (mExecutionSet)
        mContext.GetDevice()->destroy(mExecutionSet);
}

template <bool UsePipeline, bool IsCompute, class TPushConstant>
DGCSequence<UsePipeline, IsCompute, TPushConstant>&
DGCSequence<UsePipeline, IsCompute, TPushConstant>::AddPipeline(
    const char* name) {
    uint32_t idx = mPipelineNamesIdxMap.size();
    mPipelineNamesIdxMap.emplace(name, idx);
    return *this;
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
void DGCSequence<IsCompute, UsePipeline, TPushConstant>::MakeSequenceLayout(
    const char* pipelineLayoutName, bool unorderedSequence,
    bool explicitPreprocess) {
    auto pipelineLayout = mPipelineMgr.GetLayoutHandle(pipelineLayoutName);

    mLayout = MakeUnique<Type_SequenceLayout>(
        mContext, pipelineLayout, unorderedSequence, explicitPreprocess);
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
typename DGCSequence<IsCompute, UsePipeline,
                     TPushConstant>::Type_SequenceLayout&
DGCSequence<IsCompute, UsePipeline, TPushConstant>::GetLayout() const {
    return *mLayout;
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
void DGCSequence<IsCompute, UsePipeline, TPushConstant>::MakeExecutionSet(
    const char* initialPipelineName) {
    static_assert(UsePipeline);

    if (mPipelineNamesIdxMap.contains(initialPipelineName)) {
        if (mPipelineNamesIdxMap.at(initialPipelineName) != 0) {
            auto it = ::std::find_if(
                mPipelineNamesIdxMap.begin(), mPipelineNamesIdxMap.end(),
                [&](const auto& pair) { return pair.second == 0; });

            ::std::swap(it->second,
                        mPipelineNamesIdxMap.at(initialPipelineName));
        }
    } else {
        for (auto& [name, idx] : mPipelineNamesIdxMap) {
            idx += 1;
        }
        mPipelineNamesIdxMap.emplace(initialPipelineName, 0);
    }

    vk::IndirectExecutionSetPipelineInfoEXT esPipelineInfo {};
    esPipelineInfo
        .setInitialPipeline(mPipelineMgr.GetPipelineHandle(initialPipelineName))
        .setMaxPipelineCount(mMaxPipelineCount);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::ePipelines)
        .setInfo(&esPipelineInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetPipelineEXT> writeIES {};
    writeIES.reserve(mPipelineNamesIdxMap.size());

    for (const auto& [name, idx] : mPipelineNamesIdxMap) {
        vk::WriteIndirectExecutionSetPipelineEXT write {};
        write.setPipeline(mPipelineMgr.GetPipelineHandle(name.c_str()))
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetPipelineEXT(mExecutionSet,
                                                                writeIES);
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
vk::IndirectExecutionSetEXT
DGCSequence<IsCompute, UsePipeline, TPushConstant>::GetExecutionSet() const {
    return mExecutionSet;
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
Type_STLVector<typename DGCSequence<IsCompute, UsePipeline,
                                    TPushConstant>::Type_SequenceTemplate>&
DGCSequence<IsCompute, UsePipeline, TPushConstant>::GetSequenceData() {
    VE_ASSERT((UsePipeline && mExecutionSet != VK_NULL_HANDLE) || !UsePipeline,
              "Sequence Data should be set after MakeExecutionSet");

    if (mSequenceData.empty())
        mSequenceData.resize(mSequenceCount);
    return mSequenceData;
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
vk::DeviceAddress DGCSequence<IsCompute, UsePipeline,
                              TPushConstant>::GetSequenceDataBufferAddress()
    const {
    return mSequenceDataBuffer->GetDeviceAddress();
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
void DGCSequence<IsCompute, UsePipeline, TPushConstant>::Finalize() {
    GenerateSequenceDataBuffer();
}

template <bool IsCompute, bool UsePipeline, class TPushConstant>
void DGCSequence<IsCompute, UsePipeline,
                 TPushConstant>::GenerateSequenceDataBuffer() {
    VE_ASSERT(!mSequenceData.empty(), "No sequence data was set.")

    auto const sequenceDataSize =
        sizeof(Type_SequenceTemplate) * mSequenceCount;

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
