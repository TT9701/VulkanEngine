#pragma once

#include "Core/System/IDDeferredResourcePool.hpp"
#include "Core/Utilities/Camera.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Utilities/Threading/Thread.hpp"
#include "NodeProxy.hpp"

#include "JSON/NlohmannJSON_v3_11_3/json.hpp"

#include <mutex>

#include "Core/Model/ModelDataManager.h"

namespace IntelliDesign_NS::Vulkan::Core {
class DGCSeqManager;
}

namespace IntelliDesign_NS::Core::SceneGraph {

class Node;

class Scene {
    using Type_NodeMap = MemoryPool::Type_STLUnorderedMap_String<
        MemoryPool::Type_SharedPtr<Node>>;

    friend class Node;

public:
    Scene(Vulkan::Core::DGCSeqManager& seqMgr,
          Vulkan::Core::GPUGeometryDataManager& gpuGeoDataMgr,
          ModelData::ModelDataManager& modelDataMgr,
          ::std::pmr::memory_resource* pMemPool);

    /**
     *  make & add & remove node
     */

    template <class TDGCSeqTemp>
    MemoryPool::Type_SharedPtr<NodeProxy<TDGCSeqTemp>> MakeNodeProxy(
        Type_pSeqDataBufPool pool);

    template <class TDGCSeqTemp>
    bool AddNodeProxy_Async(const char* path, Type_pSeqDataBufPool pool,
                            Camera* camera = nullptr);

    bool RemoveNode_Async(const char* name);

    /**
     *  frustum culling
     */

    void CullNode(MathCore::BoundingFrustum const& frustum,
                  Vulkan::Core::RenderFrame& frame);

    /**
     *  visit nodes
     */

    MemoryPool::Type_SharedPtr<Node> GetNode(const char* name);

    MemoryPool::Type_SharedPtr<Node> GetNode(uint32_t id);

    void VisitAllNodes(::std::function<void(Node* node)> const& func);

    uint32_t GetNodeCount();

    /**
     *   JSON related
     */

    void WriteToJSON() const;

    template <class TDGCSeqTemp>
    void ReadFromJSON(const char* jsonFile, Type_pSeqDataBufPool pool);

private:
    Node MakeNodeInstance(const char* name = nullptr);

    template <class TDGCSeqTemp>
    Node& EmplaceNodeProxy(
        MemoryPool::Type_SharedPtr<NodeProxy<TDGCSeqTemp>>&& nodeProxy);

    void RemoveNode_Sync(const char* name);

    void RegisterModel(Node& node, const char* path);

    void UnregisterModel(Node& node);

private:
    ::std::pmr::memory_resource* pMemPool;

    Vulkan::Core::DGCSeqManager& mDGCSeqMgr;
    Vulkan::Core::GPUGeometryDataManager& mGPUGeoDataMgr;
    ModelData::ModelDataManager& mModelDataMgr;

    ::std::recursive_mutex mNodeMapMutex;
    Type_NodeMap mNodes;

    IDPool_Queue<uint32_t> mIDQueue {};

    using Type_TaskMap = MemoryPool::Type_STLUnorderedMap_String<
        Vulkan::Core::SharedPtr<TaskRequestHandleCoarse<void>>>;

    /**
     *  async load related
     */
    Thread mWokerThread;

    ::std::mutex mAddTaskMapMutex;
    Type_TaskMap mAddTaskMap;

    ::std::mutex mRemoveTaskSetMutex;
    MemoryPool::Type_STLUnorderedSet<MemoryPool::Type_STLString> mRemoveTaskSet;

    /**
     *  cisdi model use count. record for node nameing.
     */
    ::std::mutex mModelUsedCountMapMutex;
    MemoryPool::Type_STLUnorderedMap_String<
        MemoryPool::Type_UniquePtr<IDPool_Queue<uint32_t>>>
        mModelUsedCountMap;
};

template <class TDGCSeqTemp>
MemoryPool::Type_SharedPtr<NodeProxy<TDGCSeqTemp>> Scene::MakeNodeProxy(
    Type_pSeqDataBufPool pool) {
    return MakeShared<NodeProxy<TDGCSeqTemp>>(MakeNodeInstance(), pool);
}

template <class TDGCSeqTemp>
bool Scene::AddNodeProxy_Async(const char* path, Type_pSeqDataBufPool pool,
                               Camera* camera) {
    MemoryPool::Type_STLString pathStr {path};

    auto loadNewModel = [this, pathStr, pool, camera]() {
        auto nodeProxy = MakeNodeProxy<TDGCSeqTemp>(pool);

        RegisterModel(*nodeProxy, pathStr.c_str());

        auto const& modelData = nodeProxy->SetModel(pathStr.c_str());

        if (camera) {
            auto scale = nodeProxy->GetModelMatrixInfo().scaleVec;
            camera->AdjustPosition(modelData.boundingBox, scale);
        }

        EmplaceNodeProxy(::std::move(nodeProxy));

        {
            ::std::unique_lock lock {mAddTaskMapMutex};
            if (mAddTaskMap.contains(pathStr))
                mAddTaskMap.erase(pathStr);
        }
    };

    // loading model from disk
    {
        ::std::unique_lock lock {mAddTaskMapMutex};
        if (mAddTaskMap.contains(pathStr)) {
            // already loaded model
            mWokerThread.Submit(
                true, false, [this, pathStr, pool, camera, loadNewModel]() {
                    if (auto pModel = mModelDataMgr.Get_CISDI_3DModel_FromPath(
                            pathStr.c_str())) {

                        auto nodeProxy = MakeNodeProxy<TDGCSeqTemp>(pool);

                        if (camera)
                            camera->AdjustPosition(
                                pModel->boundingBox,
                                nodeProxy->GetModelMatrixInfo().scaleVec);

                        RegisterModel(*nodeProxy, pathStr.c_str());

                        nodeProxy->SetModel(::std::move(pModel));

                        EmplaceNodeProxy(::std::move(nodeProxy));
                    } else {
                        loadNewModel();
                    }
                });
            return true;
        }
    }

    // removing model
    {
        ::std::unique_lock lock {mRemoveTaskSetMutex};
        if (mRemoveTaskSet.contains(pathStr))
            return false;
    }

    // load a new model from disk
    auto pTask = mWokerThread.Submit(true, true, loadNewModel);

    {
        ::std::unique_lock lock {mAddTaskMapMutex};
        if (!pTask->IsReady()) {
            mAddTaskMap.emplace(pathStr, pTask);
        }
    }

    return true;
}

template <class TDGCSeqTemp>
Node& Scene::EmplaceNodeProxy(
    MemoryPool::Type_SharedPtr<NodeProxy<TDGCSeqTemp>>&& nodeProxy) {
    ::std::unique_lock lock {mNodeMapMutex};

    auto p = mNodes.emplace(nodeProxy->GetName(), ::std::move(nodeProxy));
    return *p.first->second;
}

template <class TDGCSeqTemp>
void Scene::ReadFromJSON(const char* jsonFile, Type_pSeqDataBufPool pool) {
    ::std::ifstream ifs(jsonFile);

    using json = nlohmann::json;
    json j;

    ifs >> j;

    for (uint32_t i = 0; i < j["Nodes"].size(); ++i) {
        auto name = j["Nodes"][i].at("name").get<::std::string>();

        auto path = j["Nodes"][i].at("path").get<::std::string>();

        auto refIdx = j["Nodes"][i].at("model_ref_idx").get<uint32_t>();

        ModelMatrixInfo modelMatrixInfo;

        auto scale = j["Nodes"][i].at("scale").get<::std::array<float, 3>>();
        modelMatrixInfo.scaleVec = MathCore::Float3 {scale.data()};

        auto translation =
            j["Nodes"][i].at("translation").get<::std::array<float, 3>>();
        modelMatrixInfo.translationVec = MathCore::Float3 {translation.data()};

        auto rotation =
            j["Nodes"][i].at("rotation").get<::std::array<float, 4>>();
        modelMatrixInfo.rotationQuat = MathCore::Float4 {rotation.data()};

        {
            AddNodeProxy_Async<TDGCSeqTemp>(path.c_str(), pool);

            mWokerThread.Submit(
                true, false, [this, name, refIdx, modelMatrixInfo]() {
                    auto node = GetNode(name.c_str());
                    if (node) {
                        node->SetModelMatrixInfo(modelMatrixInfo);
                        node->SetName(name.c_str());
                        node->SetRefIdx(refIdx);
                    }
                });
        }
    }
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
