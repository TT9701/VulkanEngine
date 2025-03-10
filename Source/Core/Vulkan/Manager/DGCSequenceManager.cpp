#include "DGCSequenceManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

SequenceDataBuffer::SequenceDataBuffer(VulkanContext& context,
                                       RenderResource& buffer,
                                       uint32_t seqCount, uint32_t stride)
    : context(context), buffer(buffer) {
    sequenceData.resize(seqCount * stride);
}

void SequenceDataBuffer::CreateSequenceDataBuffer() {
    VE_ASSERT(!sequenceData.empty(), "No data was set.")

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

DGCSequenceManager::DGCSequenceManager(VulkanContext& context,
                                       PipelineManager& pipelineMgr,
                                       ShaderManager& shaderMgr,
                                       RenderResourceManager& renderResMgr)
    : mContext(context),
      mPipelineMgr(pipelineMgr),
      mShaderMgr(shaderMgr),
      mRenderResMgr(renderResMgr) {}

DGCSequenceBase& DGCSequenceManager::GetSequence(const char* name) {
    auto it = mSequences.find(name);
    VE_ASSERT(it != mSequences.end(),
              "The sequence does not exist in the manager.");
    return *it->second;
}

}  // namespace IntelliDesign_NS::Vulkan::Core