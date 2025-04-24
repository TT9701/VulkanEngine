#include "Node.h"

#include "Core/Model/GPUGeometryData.h"
#include "Core/Model/GPUGeometryDataManager.h"
#include "Core/Model/ModelDataManager.h"
#include "Scene.h"

namespace IntelliDesign_NS::Core::SceneGraph {

MathCore::Mat4 ModelMatrixInfo::ToMatrix() {
    return  DirectX::XMMatrixTransformation(
        DirectX::g_XMZero,  // Scaling origin (not used, set to zero)
        DirectX::g_XMIdentityR3,  // Scaling orientation quaternion (identity)
        scaleVec.GetSIMD(),  // Scaling vector
        DirectX::g_XMZero,  // Rotation origin (not used, set to zero)
        rotationQuat.GetSIMD(),  // Rotation quaternion
        translationVec.GetSIMD()  // Translation vector
    );
}

void ModelMatrixInfo::FromMatrix(MathCore::Mat4 const& mat){
    MathCore::SIMD_Vec scale, rotation, translation;
    if (DirectX::XMMatrixDecompose(&scale, &rotation, &translation, mat.GetSIMD())) {
        scaleVec = scale;           
        rotationQuat = rotation;    
        translationVec = translation;
    } else {
        // Handle the case where decomposition fails (e.g., matrix is invalid)
        scaleVec = {1.0f, 1.0f, 1.0f};
        rotationQuat = {0.0f, 0.0f, 0.0f, 1.0f};
        translationVec = {0.0f, 0.0f, 0.0f};
    }
}


const char* Node::sEmptyNodePrefix = "Empty Node ";
::std::atomic_uint32_t Node::sEmptyNodeCount = 0;

Node::Node(::std::pmr::memory_resource* pMemPool, const char* name,
           Scene const* pScene, uint32_t id)
    : mName(name ? name
                 : MemoryPool::Type_STLString {sEmptyNodePrefix}
                       + ::std::to_string(sEmptyNodeCount++).c_str(),
            pMemPool),
      mpScene(pScene),
      mID(id) {}

Node::Node(Node&& other) noexcept
    : mName(std::move(other.mName)),
      mpScene(std::move(other.mpScene)),
      mID(std::move(other.mID)),
      mModelData(std::move(other.mModelData)),
      mSequenceCount(other.mSequenceCount) {}

void Node::SetName(const char* name) {
    if (mName.starts_with(MemoryPool::Type_STLString {sEmptyNodePrefix}))
        --sEmptyNodeCount;

    mName = name;

    if (mRefIdx > 0)
        mName = mName + "<" + ::std::to_string(mRefIdx).c_str() + ">";
}

void Node::SetModel(
    MemoryPool::Type_SharedPtr<ModelData::CISDI_3DModel const>&& model) {
    mModelData = ::std::move(model);

    mGPUGeoData =
        mpScene->mGPUGeoDataMgr.GetGPUGeometryData(mModelData->name.c_str());

    mSequenceCount = mGPUGeoData->GetSequenceCount();

    // TODO: use model name as node name for now. Delete this line when custom node name is neccesary.
    SetName(mModelData->name.c_str());
}

ModelData::CISDI_3DModel const& Node::SetModel(const char* modelPath) {
    mModelData = mpScene->mModelDataMgr.Create_CISDI_3DModel(modelPath);
    mGPUGeoData = mpScene->mGPUGeoDataMgr.CreateGPUGeometryData(*mModelData);
    mSequenceCount = mGPUGeoData->GetSequenceCount();

    // TODO: use model name as node name for now. Delete this line when custom node name is neccesary.
    SetName(mModelData->name.c_str());

    return *mModelData;
}

void Node::SetRefIdx(uint32_t refIdx) {
    mRefIdx = refIdx;
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

uint32_t Node::GetRefIdx() const {
    return mRefIdx;
}

MemoryPool::Type_STLVector<MemoryPool::Type_STLVector<Type_CopyInfo>> const&
Node::GetCopyInfos() const {
    return {};
}

uint32_t Node::GetID() const {
    return mID;
}

ModelMatrixInfo Node::GetModelMatrixInfo() const {
    return mModelMatrixInfo;
}

void Node::SetModelMatrixInfo(ModelMatrixInfo const& info) {
    mModelMatrixInfo = info;
}

}  // namespace IntelliDesign_NS::Core::SceneGraph