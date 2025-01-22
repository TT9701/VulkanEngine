#pragma once

#include "Source/Common/Common.h"

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
}

namespace IntelliDesign_NS::ModelImporter {
namespace FBXSDK {

[[nodiscard]] IntelliDesign_NS::ModelData::CISDI_3DModel Convert(
    const char* path, bool flipYZ,
    ModelData::Type_STLVector<ModelData::InternalMeshData>& tmpVertices,
    ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>& outIndices,
    ::std::pmr::memory_resource* pMemPool);

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
