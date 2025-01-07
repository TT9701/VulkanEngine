#pragma once

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
}

namespace IntelliDesign_NS::ModelImporter {
namespace Assimp {

[[nodiscard]] IntelliDesign_NS::ModelData::CISDI_3DModel Convert(
    const char* path, bool flipYZ);

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
