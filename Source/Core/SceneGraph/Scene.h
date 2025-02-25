#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Core::SceneGraph {

class Node;

class Scene {
    using Type_NodeMap = MemoryPool::Type_STLUnorderedMap_String<
        MemoryPool::Type_UniquePtr<Node>>;

public:
    explicit Scene(::std::pmr::memory_resource* pMemPool);

    Node& AddNode(MemoryPool::Type_UniquePtr<Node>&& node);

    Node& AddNode(const char* name);

    Node const& GetNode(const char* name) const;

    Node& GetNode(const char* name);

    Type_NodeMap const& GetAllNodes() const;

    void RemoveNode(const char* name);

private:
    ::std::pmr::memory_resource* pMemPool;

    Type_NodeMap mNodes;
};

}  // namespace IntelliDesign_NS::Core::SceneGraph
