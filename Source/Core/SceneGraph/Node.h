#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
class ModelDataManager;
}  // namespace IntelliDesign_NS::ModelData

namespace IntelliDesign_NS::Vulkan::Core {
class GPUGeometryDataManager;
}

namespace IntelliDesign_NS::Core::SceneGraph {

class Node {
public:
    Node(::std::pmr::memory_resource* pMemPool, const char* name);

    void SetModel(
        MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model);

    ModelData::CISDI_3DModel const& SetModel(
        const char* modelPath, Vulkan::Core::GPUGeometryDataManager& geoDataMgr,
        ModelData::ModelDataManager& modelDataMgr);

    ModelData::CISDI_3DModel const& GetModel() const;

    MemoryPool::Type_STLString const& GetName() const;

private:
    MemoryPool::Type_STLString mName;

    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const> mModelData {
        nullptr};
};

}  // namespace IntelliDesign_NS::Core::SceneGraph