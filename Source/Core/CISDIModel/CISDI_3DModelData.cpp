#include "CISDI_3DModelData.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include "Assimp_Importer.h"
#include "FBX_Importer.h"

namespace IntelliDesign_NS::ModelData {

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

void RemapIndex(CISDI_3DModel::Mesh& mesh,
                Type_STLVector<uint32_t>& tmpIndices) {
    size_t vertexCount = mesh.header.vertexCount;
    size_t indexCount = tmpIndices.empty() ? vertexCount : tmpIndices.size();

    // TODO: other attributes

    Type_STLVector<meshopt_Stream> streams = {
        {mesh.vertices.positions.data(), sizeof(mesh.vertices.positions[0]),
         sizeof(mesh.vertices.positions[0])},
        {mesh.vertices.normals.data(), sizeof(mesh.vertices.normals[0]),
         sizeof(mesh.vertices.normals[0])},
        {mesh.vertices.uvs.data(), sizeof(mesh.vertices.uvs[0]),
         sizeof(mesh.vertices.uvs[0])}};

    Type_STLVector<uint32_t> remap(indexCount);
    vertexCount = meshopt_generateVertexRemapMulti(
        remap.data(), tmpIndices.empty() ? nullptr : tmpIndices.data(),
        indexCount, vertexCount, streams.data(), streams.size());

    decltype(mesh.vertices.positions) optimizedVertexPositions(vertexCount);
    decltype(mesh.vertices.normals) optimizedVertexNormals(vertexCount);
    decltype(mesh.vertices.uvs) optimizedVertexUVs(vertexCount);

    Type_STLVector<uint32_t> optimizedIndices(indexCount);

    meshopt_remapVertexBuffer(optimizedVertexPositions.data(),
                              mesh.vertices.positions.data(), indexCount,
                              sizeof(mesh.vertices.positions[0]), remap.data());

    meshopt_remapVertexBuffer(optimizedVertexNormals.data(),
                              mesh.vertices.normals.data(), indexCount,
                              sizeof(mesh.vertices.normals[0]), remap.data());

    meshopt_remapVertexBuffer(optimizedVertexUVs.data(),
                              mesh.vertices.uvs.data(), indexCount,
                              sizeof(mesh.vertices.uvs[0]), remap.data());

    meshopt_remapIndexBuffer(optimizedIndices.data(), tmpIndices.data(),
                             indexCount, remap.data());

    mesh.header.vertexCount = vertexCount;

    mesh.vertices.positions = std::move(optimizedVertexPositions);
    mesh.vertices.normals = std::move(optimizedVertexNormals);
    mesh.vertices.uvs = std::move(optimizedVertexUVs);
    tmpIndices = std::move(optimizedIndices);
}

void RemapIndex(CISDI_3DModel& data,
                Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    if (tmpIndices.empty()) {
        tmpIndices.resize(data.meshes.size());
    } else {
        assert(tmpIndices.size() == data.meshes.size());
    }

    for (uint32_t i = 0; i < data.meshes.size(); ++i) {
        RemapIndex(data.meshes[i], tmpIndices[i]);
    }
}

void OptimizeMesh(CISDI_3DModel::Mesh& mesh,
                  Type_STLVector<uint32_t>& tmpIndices) {
    auto indexCount = tmpIndices.size();
    auto vertexCount = mesh.header.vertexCount;

    const auto indices = tmpIndices.data();

    meshopt_optimizeVertexCache(indices, indices, indexCount, vertexCount);

    meshopt_optimizeOverdraw(indices, indices, indexCount,
                             (const float*)mesh.vertices.positions.data(),
                             vertexCount, sizeof(mesh.vertices.positions[0]),
                             1.05f);

    Type_STLVector<uint32_t> remap(vertexCount);
    meshopt_optimizeVertexFetchRemap(remap.data(), indices, indexCount,
                                     vertexCount);

    meshopt_remapVertexBuffer(mesh.vertices.positions.data(),
                              mesh.vertices.positions.data(), vertexCount,
                              sizeof(mesh.vertices.positions[0]), remap.data());

    meshopt_remapVertexBuffer(mesh.vertices.normals.data(),
                              mesh.vertices.normals.data(), vertexCount,
                              sizeof(mesh.vertices.normals[0]), remap.data());

    meshopt_remapVertexBuffer(mesh.vertices.uvs.data(),
                              mesh.vertices.uvs.data(), vertexCount,
                              sizeof(mesh.vertices.uvs[0]), remap.data());

    meshopt_remapIndexBuffer(indices, indices, indexCount, remap.data());
}

void OptimizeData(CISDI_3DModel& data,
                  Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    assert(data.meshes.size() == tmpIndices.size());
    for (uint32_t i = 0; i < data.meshes.size(); ++i) {
        OptimizeMesh(data.meshes[i], tmpIndices[i]);
    }
}

void BuildMeshlet(CISDI_3DModel::Mesh& mesh,
                  Type_STLVector<uint32_t>& tmpIndices) {
    size_t maxMeshlets =
        meshopt_buildMeshletsBound(tmpIndices.size(), MESHLET_MAX_VERTEX_COUNT,
                                   MESHLET_MAX_TRIANGLE_COUNT);

    mesh.meshlets.resize(maxMeshlets);
    mesh.meshletVertices.resize(maxMeshlets * MESHLET_MAX_VERTEX_COUNT);
    mesh.meshletTriangles.resize(maxMeshlets * MESHLET_MAX_TRIANGLE_COUNT * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        mesh.meshlets.data(), mesh.meshletVertices.data(),
        mesh.meshletTriangles.data(), tmpIndices.data(), tmpIndices.size(),
        (const float*)mesh.vertices.positions.data(), mesh.header.vertexCount,
        sizeof(mesh.vertices.positions[0]), MESHLET_MAX_VERTEX_COUNT,
        MESHLET_MAX_TRIANGLE_COUNT, 0.0f);

    const meshopt_Meshlet& last = mesh.meshlets[meshletCount - 1];

    mesh.meshletVertices.resize(last.vertex_offset + last.vertex_count);
    mesh.meshletTriangles.resize(last.triangle_offset
                                 + ((last.triangle_count * 3 + 3) & ~3));
    mesh.meshlets.resize(meshletCount);

    for (auto& meshlet : mesh.meshlets) {
        meshopt_optimizeMeshlet(&mesh.meshletVertices[meshlet.vertex_offset],
                                &mesh.meshletTriangles[meshlet.triangle_offset],
                                meshlet.triangle_count, meshlet.vertex_count);
    }

    mesh.header.meshletCount = mesh.meshlets.size();
    mesh.header.meshletVertexCount = mesh.meshletVertices.size();
    mesh.header.meshletTriangleCount = mesh.meshletTriangles.size();
}

void BuildMeshletDatas(CISDI_3DModel& data,
                       Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    for (uint32_t i = 0; i < data.meshes.size(); ++i) {
        BuildMeshlet(data.meshes[i], tmpIndices[i]);
    }
}

void WriteDataHeader(std::ofstream& ofs, CISDI_3DModel::Header header) {
    ofs.write((char*)&header, sizeof(header));
}

void WriteString(std::ofstream& ofs, const char* str) {
    size_t nameLen = strlen(str);
    ofs.write((char*)&nameLen, sizeof(nameLen));
    ofs.write(str, nameLen);
}

void WriteNodes(std::ofstream& ofs,
                Type_STLVector<CISDI_3DModel::Node> const& nodes) {
    for (auto const& node : nodes) {
        WriteString(ofs, node.name.c_str());
        auto offset = strlen(node.name.c_str());
        ofs.write((char*)&node.meshIdx, sizeof(uint32_t) * 4);
        if (node.childCount > 0)
            ofs.write((char*)node.childrenIdx.data(),
                      sizeof(node.childrenIdx[0]) * node.childCount);
    }
}

void WriteMeshHeader(std::ofstream& ofs,
                     CISDI_3DModel::Mesh::MeshHeader const& meshHeader) {
    ofs.write((char*)&meshHeader, sizeof(meshHeader));
}

void WriteMeshes(std::ofstream& ofs,
                 Type_STLVector<CISDI_3DModel::Mesh> const& meshes) {
    for (auto const& mesh : meshes) {
        WriteMeshHeader(ofs, mesh.header);

        // positions
        ofs.write((char*)mesh.vertices.positions.data(),
                  sizeof(mesh.vertices.positions[0]) * mesh.header.vertexCount);

        // normals
        ofs.write((char*)mesh.vertices.normals.data(),
                  sizeof(mesh.vertices.normals[0]) * mesh.header.vertexCount);

        // uvs
        ofs.write((char*)mesh.vertices.uvs.data(),
                  sizeof(mesh.vertices.uvs[0]) * mesh.header.vertexCount);

        // TODO: other attributes

        // meshlet datas
        if (mesh.header.meshletCount > 0) {
            ofs.write((char*)mesh.meshlets.data(),
                      sizeof(mesh.meshlets[0]) * mesh.header.meshletCount);
        }

        if (mesh.header.meshletVertexCount > 0) {
            ofs.write((char*)mesh.meshletVertices.data(),
                      sizeof(mesh.meshletVertices[0])
                          * mesh.header.meshletVertexCount);
        }

        if (mesh.header.meshletTriangleCount > 0) {
            ofs.write((char*)mesh.meshletTriangles.data(),
                      sizeof(mesh.meshletTriangles[0])
                          * mesh.header.meshletTriangleCount);
        }
    }
}

void WriteMaterial(std::ofstream& ofs,
                   Type_STLVector<CISDI_3DModel::Material> const& materials) {
    for (auto const& material : materials) {
        WriteString(ofs, material.name.c_str());
        ofs.write((char*)&material.ambient,
                  sizeof(material.ambient) * 3 + sizeof(material.opacity));
    }
}

void WriteFile(const char* outputPath, CISDI_3DModel const& data) {
    ::std::ofstream out(outputPath, ::std::ios::out | ::std::ios::binary);

    if (!out.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("fail to open file: ") + outputPath).c_str());
    }

    WriteDataHeader(out, data.header);
    WriteString(out, data.name.c_str());
    WriteNodes(out, data.nodes);
    WriteMeshes(out, data.meshes);
    WriteMaterial(out, data.materials);
}

CISDI_3DModel Convert(const char* path, bool flipYZ, const char* output) {
    auto outputPath = ProcessOutputPath(path, output);

    CISDI_3DModel data {};

    // process input model file
    {
        Type_STLVector<Type_STLVector<uint32_t>> tmpIndices {};

        if (::std::filesystem::path(path).extension() == ".fbx"
            || ::std::filesystem::path(path).extension() == ".FBX") {
            data = IntelliDesign_NS::ModelImporter::FBXSDK::Convert(
                path, flipYZ, tmpIndices);
            RemapIndex(data, tmpIndices);
        } else {
            data = IntelliDesign_NS::ModelImporter::Assimp::Convert(
                path, flipYZ, tmpIndices);
        }

        OptimizeData(data, tmpIndices);

        BuildMeshletDatas(data, tmpIndices);
    }

    WriteFile(outputPath.c_str(), data);

    return data;
}

void ReadString(::std::ifstream& in, Type_STLString& str) {
    size_t nameLen;
    in.read((char*)&nameLen, sizeof(nameLen));
    str.resize(nameLen);
    in.read(str.data(), nameLen);
}

CISDI_3DModel Load(const char* path) {
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
            (Type_STLString("ERROR::CISDI_3DModel::Convert ") + path).c_str());
    }

    // TODO: Version Check

    // read name
    ReadString(in, data.name);

    // read node
    data.nodes.resize(data.header.nodeCount);
    for (auto& node : data.nodes) {
        ReadString(in, node.name);
        in.read((char*)&node.meshIdx, sizeof(uint32_t) * 4);
        if (node.childCount > 0) {
            node.childrenIdx.resize(node.childCount);
            in.read((char*)node.childrenIdx.data(),
                    sizeof(node.childrenIdx[0]) * node.childCount);
        }
    }

    // read mesh
    data.meshes.resize(data.header.meshCount);
    for (auto& mesh : data.meshes) {
        in.read((char*)&mesh.header, sizeof(mesh.header));

        mesh.vertices.positions.resize(mesh.header.vertexCount);
        in.read((char*)mesh.vertices.positions.data(),
                mesh.header.vertexCount * sizeof(mesh.vertices.positions[0]));

        mesh.vertices.normals.resize(mesh.header.vertexCount);
        in.read((char*)mesh.vertices.normals.data(),
                mesh.header.vertexCount * sizeof(mesh.vertices.normals[0]));

        mesh.vertices.uvs.resize(mesh.header.vertexCount);
        in.read((char*)mesh.vertices.uvs.data(),
                mesh.header.vertexCount * sizeof(mesh.vertices.uvs[0]));

        // TODO: other attributes

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

    // read material
    data.materials.resize(data.header.materialCount);
    for (auto& mat : data.materials) {
        ReadString(in, mat.name);
        in.read((char*)&mat.ambient,
                sizeof(mat.ambient) * 3 + sizeof(mat.opacity));
    }

    return data;
}

}  // namespace IntelliDesign_NS::ModelData