#pragma once

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/DGCSeqDataBufPoolResource.h"

namespace IntelliDesign_NS::Vulkan::Core {
class GPUGeometryDataManager;
class GPUGeometryData;
class RenderFrame;
}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
class ModelDataManager;
}  // namespace IntelliDesign_NS::ModelData

namespace IntelliDesign_NS::Core::SceneGraph {

struct ModelMatrixInfo {
    ::std::array<float, 3> scaleVec {1.0f, 1.0f, 1.0f};
    ::std::array<float, 3> transformVec {0.0f, 0.0f, 0.0f};
    ::std::array<float, 4> rotationQuat {0.0f, 0.0f, 0.0f, 1.0f};
};

class Scene;

using Type_CopyInfo =
    Vulkan::Core::DGCSeqDataBufPoolResource::ResourceHandle::CopyInfo;

class Node {
    static const char* sEmptyNodePrefix;
    static ::std::atomic_uint32_t sEmptyNodeCount;

public:
    Node(::std::pmr::memory_resource* pMemPool, const char* name,
         Scene const* pScene, uint32_t id);

    Node(Node&& other) noexcept;

    virtual ~Node() = default;

    void SetName(const char* name);

    void SetModel(
        MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model);

    ModelData::CISDI_3DModel const& SetModel(const char* modelPath);

    void SetRefIdx(uint32_t refIdx);

    ModelData::CISDI_3DModel const& GetModel() const;

    MemoryPool::Type_SharedPtr<Vulkan::Core::GPUGeometryData> GetGPUGeoDataRef()
        const;

    MemoryPool::Type_STLString const& GetName() const;

    uint32_t GetRefIdx() const;

    virtual void RequestSeqBufIDs() {}

    virtual void RetrieveIDs() {}

    virtual void UploadSeqBuf(Vulkan::Core::RenderFrame& frame) {}

    virtual MemoryPool::Type_STLVector<
        MemoryPool::Type_STLVector<Type_CopyInfo>> const&
    GetCopyInfos() const;

    virtual MemoryPool::Type_STLVector<uint64_t> const& GetSeqBufIDs() const {
        return {};
    }

    uint32_t GetID() const;

    ModelMatrixInfo GetModelMatrixInfo() const;

    void SetModelMatrixInfo(ModelMatrixInfo const& info);

protected:
    MemoryPool::Type_STLString mName;
    Scene const* mpScene;
    uint32_t mID;

    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const> mModelData {
        nullptr};
    uint32_t mRefIdx {0};

    MemoryPool::Type_SharedPtr<Vulkan::Core::GPUGeometryData> mGPUGeoData {
        nullptr};

    uint32_t mSequenceCount {0};

    ModelMatrixInfo mModelMatrixInfo;
};

}  // namespace IntelliDesign_NS::Core::SceneGraph