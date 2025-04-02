#include "DGCSeqDataBufPoolResource.h"

#include "Buffer.h"
#include "Core/Vulkan/Manager/DGCSeqManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

DGCSeqDataBufPoolResource::ResourceHandle::ResourceHandle(
    void* p, size_t size, uint64_t id, DGCSeqDataBufPoolResource* poolRes)
    : ptr(p), size(size), id(id), pPoolResource(poolRes) {}

DGCSeqDataBufPoolResource::ResourceHandle::~ResourceHandle() {}

DGCSeqDataBufPoolResource::ResourceHandle::CopyInfo
DGCSeqDataBufPoolResource::ResourceHandle::GetCopyInfo(uint32_t idx) const {
    return {pPoolResource->mStagingBufNames[idx].c_str(),
            pPoolResource->mBufName.c_str(),
            vk::BufferCopy2 {0, 0, pPoolResource->mSeqStride}};
}

void* DGCSeqDataBufPoolResource::ResourceHandle::GetStagingMappedPtr(
    uint32_t idx) const {
    auto& staging =
        pPoolResource->mResMgr[pPoolResource->mStagingBufNames[idx].c_str()];
    return staging.GetBufferMappedPtr();
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
    Type_STLString stagingBufName = seqName;

    mBufName +=
        Type_STLString {"_data_buf_"} + ::std::to_string(resIdx).c_str();
    stagingBufName += Type_STLString {"_data_staging_buf_"}
                    + ::std::to_string(resIdx).c_str() + "_";

    for (uint32_t i = 0; i < 3; ++i) {
        mStagingBufNames.push_back(stagingBufName
                                   + ::std::to_string(i).c_str());
        mResMgr.CreateBuffer(mStagingBufNames.back().c_str(),
                             seqCount * seqStride,
                             vk::BufferUsageFlagBits::eTransferSrc
                                 | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                 | vk::BufferUsageFlagBits::eStorageBuffer,
                             Buffer::MemoryType::Staging, seqStride);
    }

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
    std::optional<DGCSeqDataBufPoolResource::_Type_Resource_> res;
    if (id >= mResources.size())
        return ::std::nullopt;
    res.emplace(&mResources[id * mSeqStride], mSeqStride, id, this);
    return res;
}

Type_STLString const& DGCSeqDataBufPoolResource::GetName() const {
    return mBufName;
}

}  // namespace IntelliDesign_NS::Vulkan::Core
