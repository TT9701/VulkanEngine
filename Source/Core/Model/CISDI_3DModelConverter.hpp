#pragma once

#include "Mesh.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class CISDI_3DModelDataConverter {
public:
    CISDI_3DModelDataConverter(const char* path,
                               const char* outputDirectory = "",
                               bool flipYZ = true);

    void Execute();

    static Type_STLVector<Mesh> LoadCISDIModelData(const char* path);

private:
    bool mFlipYZ;

    Type_STLString mPath;
    Type_STLString mOutputDirectory;
};

}  // namespace IntelliDesign_NS::Vulkan::Core