#include "CISDI_3DModelConverter.hpp"

#include "CISDI_3DModelData.hpp"

CISDI_3DModelDataConverter::CISDI_3DModelDataConverter(
    const char* path, const char* outputDirectory, bool flipYZ)
    : mFlipYZ(flipYZ), mPath(path), mOutputDirectory(outputDirectory) {}

void CISDI_3DModelDataConverter::Execute() {
    if (mOutputDirectory.empty()) {
        CISDI_3DModelData::Convert(mPath.c_str(), mFlipYZ, nullptr);
    } else {
        CISDI_3DModelData::Convert(mPath.c_str(), mFlipYZ,
                                   mOutputDirectory.c_str());
    }
}

std::vector<Mesh> CISDI_3DModelDataConverter::LoadCISDIModelData(
    std::string const& path) {
    auto data = CISDI_3DModelData::Load(path.c_str());

    ::std::vector<Mesh> meshes;
    meshes.reserve(data.header.meshCount);
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        ::std::vector<Vertex> vertices;
        vertices.reserve(data.meshes[i].header.vertexCount);
        for (uint32_t j = 0; j < data.meshes[i].header.vertexCount; ++j) {
            Vertex v {};
            v.position.x = data.meshes[i].vertices.positions[j].x;
            v.position.y = data.meshes[i].vertices.positions[j].y;
            v.position.z = data.meshes[i].vertices.positions[j].z;
            v.normal.x   = data.meshes[i].vertices.normals[j].x;
            v.normal.y   = data.meshes[i].vertices.normals[j].y;
            v.normal.z   = data.meshes[i].vertices.normals[j].z;

            vertices.push_back(v);
        }

        meshes.emplace_back(std::move(vertices),
                            std::move(data.meshes[i].indices));
    }

    return meshes;
}