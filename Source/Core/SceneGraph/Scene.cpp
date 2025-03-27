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
        auto dataName = mNodes.at(name)->GetModel().name;

        mNodes.erase(name);

        mModelDataMgr.Remove_CISDI_3DModel(dataName.c_str());
        mGPUGeoDataMgr.RemoveGPUGeometryData(dataName.c_str());
    } else {
        DBG_LOG_INFO("Scene::RemoveNode: Node %s not found.", name);
    }
}

void Scene::CullNode(MathCore::BoundingFrustum const& frustum) {
    for (auto const& [name, node] : mNodes) {
        auto const& bb = node->GetModel().boundingBox;
        if (frustum.Contains(bb) != MathCore::ContainmentType::DISJOINT) {
            mInFrustumNodes.push_back(node.get());
        }
    }
}

MemoryPool::Type_STLVector<Node*> const& Scene::GetInFrustumNodes() const {
    return mInFrustumNodes;
}

void Scene::ClearInFrustumNodes() {
    mInFrustumNodes.clear();
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
