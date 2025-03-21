#include "DGCSeqDataBufPoolResource.h"

#include "Core/Vulkan/Manager/DGCSeqManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

DGCSeqDataBufPoolResource::ResourceHandle::ResourceHandle(
    void* p, size_t size, uint64_t id, DGCSeqDataBufPoolResource* poolRes)
    : ptr(p), size(size), id(id), pPoolResource(poolRes) {}

DGCSeqDataBufPoolResource::ResourceHandle::~ResourceHandle() {
    auto staging = pPoolResource->mContext.CreateStagingBuffer("", size);

    void* data = staging->GetMapPtr();
    memcpy(data, ptr, size);

    {
        auto cmd = pPoolResource->mContext.CreateCmdBufToBegin(
            pPoolResource->mContext.GetQueue(QueueType::Transfer));
        vk::BufferCopy cmdBufCopy {};
        cmdBufCopy.setSize(size).setDstOffset(id * pPoolResource->mSeqStride);
        cmd->copyBuffer(staging->GetHandle(), pPoolResource->mHandle,
                        cmdBufCopy);
    }
}

DGCSeqDataBufPoolResource::DGCSeqDataBufPoolResource(
    uint32_t seqCount, uint32_t resIdx, VulkanContext& context,
    RenderResourceManager& resMgr, DGCSeqManager& seqMgr, const char* seqName,
    uint32_t seqStride)
    : mContext(context),
      mResMgr(resMgr),
      mSeqMgr(seqMgr),
      mSeqStride(seqStride) {
    mResources.resize(seqCount * seqStride);

    mBufName = seqName;
    mBufName +=
        Type_STLString {"_data_buf_"} + ::std::to_string(resIdx).c_str();

    auto& buf =
        mResMgr.CreateBuffer(mBufName.c_str(), seqCount * seqStride,
                             vk::BufferUsageFlagBits::eStorageBuffer
                                 | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                 | vk::BufferUsageFlagBits::eTransferDst,
                             Buffer::MemoryType::DeviceLocal, seqStride);

    buf.SetBufferDGCSequence(mSeqMgr.GetSequenceRef(seqName));

    mHandle = buf.GetBufferHandle();
}

std::optional<DGCSeqDataBufPoolResource::_Type_Resource_>
DGCSeqDataBufPoolResource::_Get_Resource_(size_t id) {
    if (id >= mResources.size())
        return ::std::nullopt;
    return _Type_Resource_ {&mResources[id * mSeqStride], mSeqStride, id, this};
}

Type_STLString const& DGCSeqDataBufPoolResource::GetName() const {
    return mBufName;
}

}  // namespace IntelliDesign_NS::Vulkan::Core
