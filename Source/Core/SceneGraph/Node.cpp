#include "Node.h"

#include <filesystem>

#include "Core/Model/GPUGeometryData.h"
#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"
#include "Scene.h"

namespace IntelliDesign_NS::Core::SceneGraph {
Node::Node(::std::pmr::memory_resource* pMemPool, const char* name,
           Scene const* pScene)
    : mName(name, pMemPool), mpScene(pScene) {}

Node::Node(Node&& other) noexcept
    : mName(std::move(other.mName)),
      mpScene(std::move(other.mpScene)),
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

}  // namespace IntelliDesign_NS::Core::SceneGraph