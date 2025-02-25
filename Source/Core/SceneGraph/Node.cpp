#include "Node.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"

namespace IntelliDesign_NS::Core::SceneGraph {
Node::Node(::std::pmr::memory_resource* pMemPool, const char* name)
    : mName(name, pMemPool) {}

void Node::SetModel(
    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model) {
    mModelData = ::std::move(model);
}

ModelData::CISDI_3DModel const& Node::SetModel(
    const char* modelPath, Vulkan::Core::GPUGeometryDataManager& geoDataMgr,
    ModelData::ModelDataManager& modelDataMgr) {
    mModelData = modelDataMgr.Create_CISDI_3DModel(modelPath);

    geoDataMgr.CreateGPUGeometryData(*mModelData);

    return *mModelData;
}

ModelData::CISDI_3DModel const& Node::GetModel() const {
    return *mModelData;
}

MemoryPool::Type_STLString const& Node::GetName() const {
    return mName;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph