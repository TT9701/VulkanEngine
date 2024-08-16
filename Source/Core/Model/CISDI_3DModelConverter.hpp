#pragma once

#include "Mesh.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class CISDI_3DModelDataConverter {
public:
    CISDI_3DModelDataConverter(const char* path,
                               const char* outputDirectory = "",
                               bool flipYZ = true);

    void Execute();

    static ::std::vector<Mesh> LoadCISDIModelData(const char* path);

private:
    bool mFlipYZ;

    ::std::string mPath;
    ::std::string mOutputDirectory;
};

}  // namespace IntelliDesign_NS::Vulkan::Core