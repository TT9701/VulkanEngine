#include "Scene.h"

#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"
#include "Core/Utilities/Logger.h"

using namespace IntelliDesign_NS::Core::MemoryPool;

namespace IntelliDesign_NS::Core::SceneGraph {

Scene::Scene(Vulkan::Core::DGCSeqManager& seqMgr,
             Vulkan::Core::GPUGeometryDataManager& gpuGeoDataMgr,
             ModelData::ModelDataManager& modelDataMgr,
             std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool),
      mDGCSeqMgr(seqMgr),
      mGPUGeoDataMgr(gpuGeoDataMgr),
      mModelDataMgr(modelDataMgr),
      mNodes(pMemPool),
      mWokerThread(pMemPool),
      mAddTaskMap(pMemPool),
      mRemoveTaskSet(pMemPool) {}

Node Scene::MakeNodeInstance(const char* name) {
    return Node {pMemPool, name, this, mIDQueue.RequestID()};
}

Type_SharedPtr<Node> Scene::GetNode(const char* name) {
    ::std::unique_lock lock {mNodeMapMutex};

    if (!mNodes.contains(name))
        return nullptr;

    return mNodes.at(name);
}

Type_SharedPtr<Node> Scene::GetNode(uint32_t id) {
    ::std::unique_lock lock {mNodeMapMutex};

    return nullptr;
}

void Scene::RemoveNode_Sync(const char* name) {
    ::std::unique_lock lock {mNodeMapMutex};
    if (mNodes.contains(name)) {
        auto& node = mNodes.at(name);

        auto path = mModelDataMgr.Get_CISDI_3DModel_Path(node->GetModel());

        bool deleteResource = mModelUsedCountMap.at(path)->GetUsedCount() == 1;

        mIDQueue.RetrieveID(node->GetID());

        auto dataName = node->GetModel().name;
        node->RetrieveIDs();

        UnregisterModel(*node);

        mNodes.erase(name);

        if (deleteResource) {
            mModelDataMgr.Remove_CISDI_3DModel(dataName.c_str());
            mGPUGeoDataMgr.RemoveGPUGeometryData(dataName.c_str());
        }
    } else {
        DBG_LOG_INFO("Scene::RemoveNode: Node %s not found.", name);
    }
}

bool Scene::RemoveNode_Async(const char* name) {
    auto node = GetNode(name);
    if (!node) {
        return false;
    }

    auto path = mModelDataMgr.Get_CISDI_3DModel_Path(node->GetModel());

    if (path.empty())
        return false;

    // loading this model
    {
        ::std::unique_lock lock {mAddTaskMapMutex};

        if (mAddTaskMap.contains(path))
            return false;
    }

    {
        ::std::unique_lock lock {mRemoveTaskSetMutex};

        // already removing this model
        if (mRemoveTaskSet.contains(path))
            return false;

        mRemoveTaskSet.emplace(path);
    }

    mWokerThread.Submit(true, false, [this, path, name]() {
        RemoveNode_Sync(name);

        {
            ::std::unique_lock lock {mRemoveTaskSetMutex};
            mRemoveTaskSet.erase(path);
        }
    });

    return true;
}

void Scene::CullNode(MathCore::BoundingFrustum const& frustum,
                     Vulkan::Core::RenderFrame& frame) {
    ZoneScoped;
    
    ::std::unique_lock lock {mNodeMapMutex};
    for (auto const& [name, node] : mNodes) {
        auto const& bb = node->GetModel().boundingBox;
        MathCore::BoundingBox transformedBB {};
        bb.Transform(transformedBB, node->GetModelMatrixInfo().ToMatrix().GetSIMD());
        if (frustum.Contains(transformedBB) != MathCore::ContainmentType::DISJOINT) {
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

void Scene::WriteToJSON() const {
    using json = nlohmann::json;

    json j;

    j["Nodes"] = json::array();
    for (auto const& [name, node] : mNodes) {
        json nodeJson;
        nodeJson["name"] = name.c_str();

        auto path = mModelDataMgr.Get_CISDI_3DModel_Path(node->GetModel());
        nodeJson["path"] = path.c_str();

        nodeJson["model_ref_idx"] = node->GetRefIdx();

        auto modelMatrixInfo = node->GetModelMatrixInfo();

        ::std::array<float, 3> scaleVec{modelMatrixInfo.scaleVec.x,
                                        modelMatrixInfo.scaleVec.y,
                                        modelMatrixInfo.scaleVec.z};
        nodeJson["scale"] = scaleVec;

        ::std::array<float, 3> translationVec{
            modelMatrixInfo.translationVec.x, modelMatrixInfo.translationVec.y,
            modelMatrixInfo.translationVec.z};
        nodeJson["translation"] = translationVec;
        ::std::array<float, 4> rotationQuat{
            modelMatrixInfo.rotationQuat.x, modelMatrixInfo.rotationQuat.y,
            modelMatrixInfo.rotationQuat.z, modelMatrixInfo.rotationQuat.w};
        nodeJson["rotation"] = rotationQuat;

        j["Nodes"].push_back(nodeJson);
    }

    ::std::ofstream ofs("scene_data.json");

    ofs << j;

    ofs.close();
}

void Scene::RegisterModel(Node& node, const char* path) {
    ::std::unique_lock lock {mModelUsedCountMapMutex};

    if (mModelUsedCountMap.contains(path)) {
        auto& pool = mModelUsedCountMap.at(path);
        node.SetRefIdx(pool->RequestID());
    } else {
        auto [it, success] = mModelUsedCountMap.emplace(
            path, MemoryPool::New_Unique<IDPool_Queue<uint32_t>>(pMemPool));
        if (!success) {
            throw ::std::runtime_error("RegisterModel: failed to emplace.");
        }
        auto& pool = it->second;
        node.SetRefIdx(pool->RequestID());
    }
}

void Scene::UnregisterModel(Node& node) {
    ::std::unique_lock lock {mModelUsedCountMapMutex};

    auto path =
        mModelDataMgr.Get_CISDI_3DModel_Path(node.GetModel().name.c_str());

    if (mModelUsedCountMap.contains(path)) {
        auto& pool = mModelUsedCountMap.at(path);
        pool->RetrieveID(node.GetRefIdx());

        if (pool->GetUsedCount() == 0) {
            mModelUsedCountMap.erase(path);
        }
    }
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
