#pragma once

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
}

namespace IntelliDesign_NS::ModelImporter {
namespace FBXSDK {

[[nodiscard]] IntelliDesign_NS::ModelData::CISDI_3DModel Convert(
    const char* path, bool flipYZ);

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
