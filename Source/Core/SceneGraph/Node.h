#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {
class GPUGeometryData;
}

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
class ModelDataManager;
}  // namespace IntelliDesign_NS::ModelData

namespace IntelliDesign_NS::Vulkan::Core {
class GPUGeometryDataManager;
}

namespace IntelliDesign_NS::Core::SceneGraph {

class Scene;

class Node {
public:
    Node(::std::pmr::memory_resource* pMemPool, const char* name,
         Scene const* pScene);

    Node(Node&& other) noexcept;

    virtual ~Node() = default;

    void SetModel(
        MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model);

    virtual ModelData::CISDI_3DModel const& SetModel(const char* modelPath);

    ModelData::CISDI_3DModel const& GetModel() const;

    MemoryPool::Type_SharedPtr<Vulkan::Core::GPUGeometryData> GetGPUGeoDataRef()
        const;

    MemoryPool::Type_STLString const& GetName() const;

    virtual void RequestSeqBufIDs() {}

    virtual void RetrieveIDs() {}

    virtual void UploadSeqBuf() {}

protected:
    MemoryPool::Type_STLString mName;
    Scene const* mpScene;

    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const> mModelData {
        nullptr};
    MemoryPool::Type_SharedPtr<Vulkan::Core::GPUGeometryData> mGPUGeoData {
        nullptr};

    uint32_t mSequenceCount {0};
};

}  // namespace IntelliDesign_NS::Core::SceneGraph