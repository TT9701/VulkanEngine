#include "DGCSeqManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

DGCSeqDataBuffer::DGCSeqDataBuffer(VulkanContext& context,
                                   RenderResource& buffer, uint32_t seqCount,
                                   uint32_t stride)
    : context(context), buffer(buffer) {
    sequenceData.resize(seqCount * stride);
}

void DGCSeqDataBuffer::CreateSequenceDataBuffer() {
    size_t const dataSize = sequenceData.size();

    auto staging = context.CreateStagingBuffer("", dataSize);

    memcpy(staging->GetMapPtr(), sequenceData.data(), dataSize);

    {
        auto cmd =
            context.CreateCmdBufToBegin(context.GetQueue(QueueType::Transfer));
        vk::BufferCopy cmdBufCopy {};
        cmdBufCopy.setSize(dataSize);
        cmd->copyBuffer(staging->GetHandle(), buffer.GetBufferHandle(),
                        cmdBufCopy);
    }
}

DGCSeqManager::DGCSeqManager(VulkanContext& context,
                             PipelineManager& pipelineMgr,
                             ShaderManager& shaderMgr,
                             RenderResourceManager& renderResMgr)
    : mContext(context),
      mPipelineMgr(pipelineMgr),
      mShaderMgr(shaderMgr),
      mRenderResMgr(renderResMgr) {}

DGCSeqBase& DGCSeqManager::GetSequence(const char* name) {
    auto it = mSequences.find(name);
    VE_ASSERT(it != mSequences.end(),
              "The sequence does not exist in the manager.");
    return *it->second;
}

SharedPtr<DGCSeqBase> DGCSeqManager::GetSequenceRef(const char* name) {
    auto it = mSequences.find(name);
    VE_ASSERT(it != mSequences.end(),
              "The sequence does not exist in the manager.");
    return it->second;
}

DGCSeqManager::Type_Sequences const& DGCSeqManager::GetAllSequences() const {
    return mSequences;
}

}  // namespace IntelliDesign_NS::Vulkan::Core