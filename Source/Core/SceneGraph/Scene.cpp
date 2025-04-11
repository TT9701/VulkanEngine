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

Node& Scene::AddNode(MemoryPool::Type_SharedPtr<Node>&& node) {
    auto name = node->GetName();

    ::std::unique_lock lock {mNodeMapMutex};

    mNodes.emplace(name, ::std::move(node));
    return *mNodes.at(name);
}

Node& Scene::AddNode(const char* name) {
    ::std::unique_lock lock {mNodeMapMutex};

    mNodes.emplace(
        name, MemoryPool::New_Shared<Node>(pMemPool, pMemPool, name, this));
    return *mNodes.at(name);
}

MemoryPool::Type_SharedPtr<Node> Scene::GetNode(const char* name) {
    ::std::unique_lock lock {mNodeMapMutex};

    return mNodes.at(name);
}

Scene::Type_NodeMap const& Scene::GetAllNodes() const {
    return mNodes;
}

void Scene::RemoveNode(const char* name) {
    ::std::unique_lock lock {mNodeMapMutex};
    if (mNodes.contains(name)) {
        auto dataName = mNodes.at(name)->GetModel().name;
        mNodes.at(name)->RetrieveIDs();
        mNodes.erase(name);

        mModelDataMgr.Remove_CISDI_3DModel(dataName.c_str());
        mGPUGeoDataMgr.RemoveGPUGeometryData(dataName.c_str());
    } else {
        DBG_LOG_INFO("Scene::RemoveNode: Node %s not found.", name);
    }
}

void Scene::CullNode(MathCore::BoundingFrustum const& frustum,
                     Vulkan::Core::RenderFrame& frame) {
    ::std::unique_lock lock {mNodeMapMutex};
    for (auto const& [name, node] : mNodes) {
        auto const& bb = node->GetModel().boundingBox;
        if (frustum.Contains(bb) != MathCore::ContainmentType::DISJOINT) {
            frame.CullRegister(node);
        }
    }
}

void Scene::VisitAllNodes(std::function<void(Node* node)> const& func) {
    ::std::unique_lock lock {mNodeMapMutex};

    for (auto const& pNode : mNodes) {
        func(pNode.second.get());
    }
}

uint32_t Scene::GetNodeCount() {
    ::std::unique_lock lock {mNodeMapMutex};

    return mNodes.size();
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
