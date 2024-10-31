#include "CISDI_3DModelData.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

namespace IntelliDesign_NS::ModelData {

uint32_t CalcMeshCount(aiNode* node) {
    uint32_t meshCount {node->mNumMeshes};
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        meshCount += CalcMeshCount(node->mChildren[i]);
    }
    return meshCount;
}

void WriteDataHeader(std::ofstream& ofs, CISDI_3DModel::Header header) {
    ofs.write((char*)&header, sizeof(header));
}

void WriteMeshHeader(std::ofstream& ofs,
                     CISDI_3DModel::Mesh::MeshHeader meshHeader) {
    ofs.write((char*)&meshHeader, sizeof(meshHeader));
}

Type_STLString ProcessOutputPath(const char* input, const char* output) {
    auto inPath = ::std::filesystem::path(input);
    ::std::filesystem::path outputPath;

    if (inPath.extension() == CISDI_3DModel_Subfix_Str) {
        return inPath.string();
    }

    if (!output) {
        outputPath = inPath.string().append(CISDI_3DModel_Subfix_Str);
    } else {
        outputPath = ::std::filesystem::canonical(output);
        if (::std::filesystem::is_directory(outputPath)) {
            outputPath = outputPath.wstring().append(L"/").append(
                inPath.wstring().append(CISDI_3DModel_Subfix_WStr));
        } else {
            throw ::std::runtime_error(
                "ERROR::CISDI_3DMODELDATA::CONVERT: Output is not a "
                "directory!");
        }
    }
    return outputPath.string();
}

void ProcessMesh(CISDI_3DModel& data, aiMesh* mesh, bool flipYZ) {
    uint32_t vertCount = mesh->mNumVertices;

    CISDI_3DModel::Mesh cisdiMesh {};
    cisdiMesh.vertices.positions.resize(vertCount);
    cisdiMesh.vertices.normals.resize(vertCount);

    // position
    for (uint32_t i = 0; i < vertCount; ++i) {
        Float4 temp {};

        temp[0] = mesh->mVertices[i].x;
        temp[1] = mesh->mVertices[i].y;
        temp[2] = mesh->mVertices[i].z;

        // TODO: temp[3] is empty for now

        if (flipYZ)
            ::std::swap(temp[1], temp[2]);

        cisdiMesh.vertices.positions[i] = temp;
    }

    // normal
    // pre calculation -> "spheremap transform"
    // wikipedia: https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    for (uint32_t i = 0; i < vertCount; ++i) {
        auto f = ::std::sqrt(2 / (1 - mesh->mNormals[i].x));

        Float2 temp {};
        temp[0] = mesh->mNormals[i].y * f;
        temp[1] = mesh->mNormals[i].z * f;

        // normal.x is runtime decoded in shader

        if (flipYZ)
            ::std::swap(temp[0], temp[1]);

        cisdiMesh.vertices.normals[i] = temp;
    }

    // texcoords
    if (mesh->HasTextureCoords(0)) {
        for (uint32_t i = 0; i < vertCount; ++i) {
            Float2 temp {};
            temp[0] = mesh->mTextureCoords[0][i].x;
            temp[1] = mesh->mTextureCoords[0][i].y;
            cisdiMesh.vertices.uvs[i] = temp;
        }
    }

    // index
    uint32_t indexCount = mesh->mNumFaces * 3;
    cisdiMesh.indices.reserve(indexCount);
    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        auto face = mesh->mFaces[i];
        for (uint32_t j = 0; j < face.mNumIndices; ++j) {
            cisdiMesh.indices.push_back(*(face.mIndices + j));
        }
    }

    cisdiMesh.header.vertexCount = vertCount;
    cisdiMesh.header.indexCount = indexCount;

    data.meshes.emplace_back(cisdiMesh);
}

void ProcessNode(CISDI_3DModel& data, aiNode* node, const aiScene* scene,
                 bool flipYZ) {
    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(data, mesh, flipYZ);
    }
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(data, node->mChildren[i], scene, flipYZ);
    }
}

void OptimizeMesh(CISDI_3DModel::Mesh& mesh) {
    size_t vertexCount = mesh.header.vertexCount;
    size_t indexCount = mesh.header.indexCount;

    // TODO: other attributes

    Type_STLVector<meshopt_Stream> streams = {
        {mesh.vertices.positions.data(), sizeof(mesh.vertices.positions[0]),
         sizeof(mesh.vertices.positions[0])},
        {mesh.vertices.normals.data(), sizeof(mesh.vertices.normals[0]),
         sizeof(mesh.vertices.normals[0])}};

    Type_STLVector<uint32_t> remap(indexCount);
    vertexCount = meshopt_generateVertexRemapMulti(
        remap.data(), mesh.indices.data(), indexCount, vertexCount,
        streams.data(), streams.size());

    Type_STLVector<Float4> optimizedVertexPositions(vertexCount);
    Type_STLVector<Float2> optimizedVertexNormals(vertexCount);

    Type_STLVector<uint32_t> optimizedIndices(indexCount);

    meshopt_remapVertexBuffer(optimizedVertexPositions.data(),
                              mesh.vertices.positions.data(), vertexCount,
                              sizeof(mesh.vertices.positions[0]), remap.data());

    meshopt_remapVertexBuffer(optimizedVertexNormals.data(),
                              mesh.vertices.normals.data(), vertexCount,
                              sizeof(mesh.vertices.normals[0]), remap.data());

    meshopt_remapIndexBuffer(optimizedIndices.data(), mesh.indices.data(),
                             indexCount, remap.data());

    mesh.header.vertexCount = vertexCount;

    meshopt_optimizeVertexCache(optimizedIndices.data(),
                                optimizedIndices.data(), indexCount,
                                vertexCount);

    meshopt_optimizeOverdraw(
        optimizedIndices.data(), optimizedIndices.data(), indexCount,
        (const float*)optimizedVertexPositions.data(), vertexCount,
        sizeof(mesh.vertices.positions[0]), 1.05f);

    mesh.vertices.positions = std::move(optimizedVertexPositions);
    mesh.vertices.normals = std::move(optimizedVertexNormals);
    mesh.indices = std::move(optimizedIndices);
}

void OptimizeData(CISDI_3DModel& data) {
    for (auto& mesh : data.meshes) {
        OptimizeMesh(mesh);
    }
}

void BuildMeshlet(CISDI_3DModel::Mesh& mesh, bool optimize) {
    size_t maxMeshlets = meshopt_buildMeshletsBound(mesh.indices.size(),
                                                    MESHLET_MAX_VERTEX_COUNT,
                                                    MESHLET_MAX_TRIANGLE_COUNT);

    mesh.meshlets.resize(maxMeshlets);
    mesh.meshletVertices.resize(maxMeshlets * MESHLET_MAX_VERTEX_COUNT);
    mesh.meshletTriangles.resize(maxMeshlets * MESHLET_MAX_TRIANGLE_COUNT * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        mesh.meshlets.data(), mesh.meshletVertices.data(),
        mesh.meshletTriangles.data(), mesh.indices.data(),
        mesh.header.indexCount, (const float*)mesh.vertices.positions.data(),
        mesh.header.vertexCount, sizeof(mesh.vertices.positions[0]),
        MESHLET_MAX_VERTEX_COUNT, MESHLET_MAX_TRIANGLE_COUNT, 0.0f);

    const meshopt_Meshlet& last = mesh.meshlets[meshletCount - 1];

    mesh.meshletVertices.resize(last.vertex_offset + last.vertex_count);
    mesh.meshletTriangles.resize(last.triangle_offset
                                 + ((last.triangle_count * 3 + 3) & ~3));
    mesh.meshlets.resize(meshletCount);

    if (optimize) {
        for (auto& meshlet : mesh.meshlets) {
            meshopt_optimizeMeshlet(
                &mesh.meshletVertices[meshlet.vertex_offset],
                &mesh.meshletTriangles[meshlet.triangle_offset],
                meshlet.triangle_count, meshlet.vertex_count);
        }
    }

    mesh.header.meshletCount = mesh.meshlets.size();
    mesh.header.meshletVertexCount = mesh.meshletVertices.size();
    mesh.header.meshletTriangleCount = mesh.meshletTriangles.size();
}

void BuildMeshletDatas(CISDI_3DModel& data, bool optimize) {
    for (auto& mesh : data.meshes) {
        BuildMeshlet(mesh, optimize);
    }
}

void WriteFile(const char* outputPath, CISDI_3DModel const& data) {
    ::std::ofstream out(outputPath, ::std::ios::out | ::std::ios::binary);

    if (!out.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("fail to open file: ") + outputPath).c_str());
    }

    WriteDataHeader(out, data.header);

    for (auto const& mesh : data.meshes) {
        WriteMeshHeader(out, mesh.header);

        // positions
        out.write((char*)mesh.vertices.positions.data(),
                  sizeof(mesh.vertices.positions[0]) * mesh.header.vertexCount);

        // normals
        out.write((char*)mesh.vertices.normals.data(),
                  sizeof(mesh.vertices.normals[0]) * mesh.header.vertexCount);

        // TODO: other attributes

        // indices
        out.write((char*)mesh.indices.data(),
                  sizeof(mesh.indices[0]) * mesh.header.indexCount);

        // meshlet datas
        if (mesh.header.meshletCount > 0) {
            out.write((char*)mesh.meshlets.data(),
                      sizeof(mesh.meshlets[0]) * mesh.header.meshletCount);
        }

        if (mesh.header.meshletVertexCount > 0) {
            out.write((char*)mesh.meshletVertices.data(),
                      sizeof(mesh.meshletVertices[0])
                          * mesh.header.meshletVertexCount);
        }

        if (mesh.header.meshletTriangleCount > 0) {
            out.write((char*)mesh.meshletTriangles.data(),
                      sizeof(mesh.meshletTriangles[0])
                          * mesh.header.meshletTriangleCount);
        }
    }
}

void CISDI_3DModel::Convert(const char* path, bool flipYZ, const char* output,
                            bool optimizeMesh, bool buildMeshlet,
                            bool optimizeMeshlet) {
    auto outputPath = ProcessOutputPath(path, output);

    CISDI_3DModel data {};

    // process input model file
    {
        Assimp::Importer importer {};

        const auto scene = importer.ReadFile(
            path, aiProcessPreset_TargetRealtime_Fast | aiProcess_OptimizeMeshes
                      | aiProcess_OptimizeGraph);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
            || !scene->mRootNode) {
            // TODO: Logging
            throw ::std::runtime_error(
                (Type_STLString("ERROR::CISDI_3DModel::Convert::ASSIMP: ")
                 + importer.GetErrorString())
                    .c_str());
        }

        data.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                       CalcMeshCount(scene->mRootNode), buildMeshlet};

        data.meshes.reserve(data.header.meshCount);

        ProcessNode(data, scene->mRootNode, scene, flipYZ);

        if (optimizeMesh) {
            OptimizeData(data);
        }

        if (buildMeshlet) {
            BuildMeshletDatas(data, optimizeMeshlet);
        }
    }

    WriteFile(outputPath.c_str(), data);
}

CISDI_3DModel CISDI_3DModel::Load(const char* path) {
    ::std::ifstream in(path, ::std::ios::binary);
    if (!in.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("fail to open file: ") + path).c_str());
    }

    CISDI_3DModel data {};

    // Header check
    in.read((char*)&data, sizeof(data.header));
    if (CISDI_3DModel_HEADER_UINT64 != data.header.header) {
        throw ::std::runtime_error(
            (Type_STLString("ERROR::CISDI_3DModel::Load ") + path).c_str());
    }

    // TODO: Version Check

    data.meshes.resize(data.header.meshCount);
    for (auto& mesh : data.meshes) {
        in.read((char*)&mesh.header, sizeof(Mesh::MeshHeader));

        mesh.vertices.positions.resize(mesh.header.vertexCount);
        in.read((char*)mesh.vertices.positions.data(),
                mesh.header.vertexCount * sizeof(mesh.vertices.positions[0]));

        mesh.vertices.normals.resize(mesh.header.vertexCount);
        in.read((char*)mesh.vertices.normals.data(),
                mesh.header.vertexCount * sizeof(mesh.vertices.normals[0]));
        // TODO: other attributes

        mesh.indices.resize(mesh.header.indexCount);
        in.read((char*)mesh.indices.data(),
                sizeof(mesh.indices[0]) * mesh.header.indexCount);

        mesh.meshlets.resize(mesh.header.meshletCount);
        in.read((char*)mesh.meshlets.data(),
                sizeof(mesh.meshlets[0]) * mesh.header.meshletCount);

        mesh.meshletVertices.resize(mesh.header.meshletVertexCount);
        in.read(
            (char*)mesh.meshletVertices.data(),
            sizeof(mesh.meshletVertices[0]) * mesh.header.meshletVertexCount);

        mesh.meshletTriangles.resize(mesh.header.meshletTriangleCount);
        in.read((char*)mesh.meshletTriangles.data(),
                sizeof(mesh.meshletTriangles[0])
                    * mesh.header.meshletTriangleCount);
    }

    return data;
}

}  // namespace IntelliDesign_NS::ModelData