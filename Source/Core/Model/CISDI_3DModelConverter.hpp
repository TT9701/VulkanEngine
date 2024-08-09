#pragma once

#include "Mesh.hpp"

class CISDI_3DModelDataConverter {
public:
    CISDI_3DModelDataConverter(const char* path,
                               const char* outputDirectory = "",
                               bool        flipYZ          = true);

    void Execute();

    static ::std::vector<Mesh> LoadCISDIModelData(::std::string const& path);

private:
    bool mFlipYZ;

    ::std::string mPath;
    ::std::string mOutputDirectory;
};