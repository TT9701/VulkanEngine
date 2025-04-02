#pragma once

#include "Core/Vulkan/Manager/RenderResourceManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class RenderResourceManager;
class DGCSeqManager;

class DGCSeqDataBufPoolResource {
public:
    class ResourceHandle {
    public:
        struct CopyInfo {
            const char* srcName;
            const char* dstName;
            vk::BufferCopy2 info;
        };

    public:
        ResourceHandle(void* p, size_t size, uint64_t id,
                       DGCSeqDataBufPoolResource* poolRes);
        ~ResourceHandle();

        CopyInfo GetCopyInfo(uint32_t idx = 0) const;

        void* GetStagingMappedPtr(uint32_t idx) const;

    public:
        void* ptr;
        size_t size;
        uint64_t id;

    private:
        DGCSeqDataBufPoolResource* pPoolResource;
    };

    friend class ResourceHandle;

public:
    using _Type_Resource_ = ResourceHandle;

    DGCSeqDataBufPoolResource(uint32_t seqCount, uint32_t resIdx,
                              VulkanContext& context,
                              RenderResourceManager& resMgr,
                              DGCSeqManager& seqMgr, const char* seqName,
                              uint32_t seqStride);

    ::std::optional<_Type_Resource_> _Get_Resource_(size_t id);

    Type_STLString const& GetName() const;

    RenderResourceManager& mResMgr;

private:
    VulkanContext& mContext;
    DGCSeqManager& mSeqMgr;

    uint32_t mSeqStride;
    Type_STLString mBufName {};
    Type_STLVector<Type_STLString> mStagingBufNames {};
    Type_STLVector<::std::byte> mResources {};
    vk::Buffer mHandle {};
};


}  // namespace IntelliDesign_NS::Vulkan::Core
