#include "Scene.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"
#include "Core/Utilities/Logger.h"

namespace IntelliDesign_NS::Core::SceneGraph {

Scene::Scene(Vulkan::Core::DGCSeqManager& seqMgr,
             Vulkan::Core::GPUGeometryDataManager& gpuGeoDataMgr,
             ModelData::ModelDataManager& modelDataMgr,
             std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool),
      mDGCSeqMgr(seqMgr),
      mGPUGeoDataMgr(gpuGeoDataMgr),
      mModelDataMgr(modelDataMgr),
      mNodes(pMemPool) {}

Node& Scene::AddNode(MemoryPool::Type_UniquePtr<Node>&& node) {
    auto name = node->GetName();
    mNodes.emplace(name, ::std::move(node));
    return *mNodes.at(name);
}

Node& Scene::AddNode(const char* name) {
    mNodes.emplace(
        name, MemoryPool::New_Unique<Node>(pMemPool, pMemPool, name, this));
    return *mNodes.at(name);
}

Node const& Scene::GetNode(const char* name) const {
    return *mNodes.at(name);
}

Node& Scene::GetNode(const char* name) {
    return *mNodes.at(name);
}

Scene::Type_NodeMap const& Scene::GetAllNodes() const {
    return mNodes;
}

void Scene::RemoveNode(const char* name) {
    if (mNodes.contains(name)) {
        auto dataName = mNodes.at(name)->GetModel().name.c_str();

        mModelDataMgr.Remove_CISDI_3DModel(dataName);
        mGPUGeoDataMgr.RemoveGPUGeometryData(dataName);

        mNodes.erase(name);
    } else {
        DBG_LOG_INFO("Scene::RemoveNode: Node %s not found.", name);
    }
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
