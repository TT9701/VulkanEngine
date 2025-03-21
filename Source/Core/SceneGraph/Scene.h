#pragma once

#include "Core/Utilities/MemoryPool.h"
#include "NodeProxy.hpp"

namespace IntelliDesign_NS::Vulkan::Core {
class DGCSeqManager;
}

namespace IntelliDesign_NS::Core::SceneGraph {

class Node;

class Scene {
    using Type_NodeMap = MemoryPool::Type_STLUnorderedMap_String<
        MemoryPool::Type_UniquePtr<Node>>;

    friend class Node;

public:
    Scene(Vulkan::Core::DGCSeqManager& seqMgr,
          Vulkan::Core::GPUGeometryDataManager& gpuGeoDataMgr,
          ModelData::ModelDataManager& modelDataMgr,
          ::std::pmr::memory_resource* pMemPool);

    Node& AddNode(MemoryPool::Type_UniquePtr<Node>&& node);

    Node& AddNode(const char* name);

    template <class TDGCSeqTemp>
    Node& AddNodeProxy(const char* name, Type_pSeqDataBufPool pool);

    Node const& GetNode(const char* name) const;

    Node& GetNode(const char* name);

    Type_NodeMap const& GetAllNodes() const;

    void RemoveNode(const char* name);

private:
    ::std::pmr::memory_resource* pMemPool;

    Vulkan::Core::DGCSeqManager& mDGCSeqMgr;
    Vulkan::Core::GPUGeometryDataManager& mGPUGeoDataMgr;
    ModelData::ModelDataManager& mModelDataMgr;

    Type_NodeMap mNodes;
};

template <class TDGCSeqTemp>
Node& Scene::AddNodeProxy(const char* name, Type_pSeqDataBufPool pool) {
    if (mNodes.contains(name)) {
        return *mNodes.at(name);
    }

    auto nodeProxy = MemoryPool::New_Unique<NodeProxy<TDGCSeqTemp>>(
        pMemPool, Node {pMemPool, name, this}, pool);

    Node& nodeRef = *nodeProxy;

    mNodes.emplace(name, std::move(nodeProxy));

    return nodeRef;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph
