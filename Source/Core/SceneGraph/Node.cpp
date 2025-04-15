#include "Node.h"

#include <filesystem>

#include "Core/Model/GPUGeometryData.h"
#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"
#include "Scene.h"

namespace IntelliDesign_NS::Core::SceneGraph {
Node::Node(::std::pmr::memory_resource* pMemPool, const char* name,
           Scene const* pScene, uint32_t id)
    : mName(name, pMemPool), mpScene(pScene), mID(id) {}

Node::Node(Node&& other) noexcept
    : mName(std::move(other.mName)),
      mpScene(std::move(other.mpScene)),
      mID(std::move(other.mID)),
      mModelData(std::move(other.mModelData)),
      mSequenceCount(other.mSequenceCount) {}

void Node::SetModel(
    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model) {
    mModelData = ::std::move(model);
}

ModelData::CISDI_3DModel const& Node::SetModel(const char* modelPath) {
    mModelData = mpScene->mModelDataMgr.Create_CISDI_3DModel(modelPath);
    mGPUGeoData = mpScene->mGPUGeoDataMgr.CreateGPUGeometryData(*mModelData);
    mSequenceCount = mGPUGeoData->GetSequenceCount();

    return *mModelData;
}

ModelData::CISDI_3DModel const& Node::GetModel() const {
    return *mModelData;
}

MemoryPool::Type_SharedPtr<Vulkan::Core::GPUGeometryData>
Node::GetGPUGeoDataRef() const {
    return mGPUGeoData;
}

MemoryPool::Type_STLString const& Node::GetName() const {
    return mName;
}

MemoryPool::Type_STLVector<MemoryPool::Type_STLVector<Type_CopyInfo>> const&
Node::GetCopyInfos() const {
    return {};
}

uint32_t Node::GetID() const {
    return mID;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph