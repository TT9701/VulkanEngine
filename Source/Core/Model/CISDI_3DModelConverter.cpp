#include "CISDI_3DModelConverter.hpp"

#include "CISDI_3DModelData.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

using IntelliDesign_NS::ModelData::CISDI_3DModel;

CISDI_3DModelDataConverter::CISDI_3DModelDataConverter(
    const char* path, const char* outputDirectory, bool flipYZ)
    : mFlipYZ(flipYZ), mPath(path), mOutputDirectory(outputDirectory) {}

void CISDI_3DModelDataConverter::Execute() {
    if (mOutputDirectory.empty()) {
        CISDI_3DModel::Convert(mPath.c_str(), mFlipYZ, nullptr);
    } else {
        CISDI_3DModel::Convert(mPath.c_str(), mFlipYZ,
                                   mOutputDirectory.c_str());
    }
}

Type_STLVector<Mesh> CISDI_3DModelDataConverter::LoadCISDIModelData(
    const char* path) {
    auto data = CISDI_3DModel::Load(path);

    Type_STLVector<Mesh> meshes;
    meshes.reserve(data.header.meshCount);
    for (auto& mesh : data.meshes) {
        Type_STLVector<Vertex> vertices;
        vertices.reserve(mesh.header.vertexCount);
        for (uint32_t j = 0; j < mesh.header.vertexCount; ++j) {
            Vertex v {};
            v.position.x = mesh.vertices.positions[j][0];
            v.position.y = mesh.vertices.positions[j][1];
            v.position.z = mesh.vertices.positions[j][2];
            v.position.w = 1.0f;
            v.normal.x = mesh.vertices.normals[j][0];
            v.normal.y = mesh.vertices.normals[j][1];
            v.normal.z = mesh.vertices.normals[j][2];

            vertices.push_back(v);
        }
        Mesh temp {vertices, mesh.indices};

        if (!mesh.meshlets.empty())
            temp.mMeshlets = mesh.meshlets;
        if (!mesh.meshletVertices.empty())
            temp.mMeshletVertices = mesh.meshletVertices;
        if (!mesh.meshletTriangles.empty())
            temp.mMeshletTriangles = mesh.meshletTriangles;

        meshes.emplace_back(temp);
    }

    return meshes;
}

}  // namespace IntelliDesign_NS::Vulkan::Core