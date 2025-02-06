#pragma once

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

template <class T>
using UniquePtr = ::std::unique_ptr<T>;

template <typename T, typename... Types>
UniquePtr<T> MakeUnique(Types&&... val) {
    return ::std::make_unique<T>(::std::forward<Types>(val)...);
}

namespace IntelliDesign_NS::ModelImporter {

namespace FBXSDK {
class Importer;
}

namespace Assimp {
class Importer;
}

class CombinedImporter {
    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;

public:
    CombinedImporter(::std::pmr::memory_resource* pMemPool, const char* path,
                     bool flipYZ, ModelData::CISDI_3DModel& outData,
                     Type_InternalMeshDatas& tmpVertices,
                     Type_Indices& outIndices);

    ~CombinedImporter();

private:
    void ProcessNode(ModelData::CISDI_3DModel& outData,
                     ModelData::CISDI_3DModel const& tmpAssimpData,
                     Type_InternalMeshDatas& tmpVertices,
                     Type_Indices& outIndices);

private:
    UniquePtr<FBXSDK::Importer> mFBXImporter;
    UniquePtr<Assimp::Importer> mAssimpImporter;
};

}  // namespace IntelliDesign_NS::ModelImporter