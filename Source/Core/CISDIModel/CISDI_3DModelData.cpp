#include "CISDI_3DModelData.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <meshoptimizer.h>

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

void RemapIndex(InternalMeshData& tmpMeshVertices,
                Type_STLVector<uint32_t>& tmpIndices) {
    auto& positions = tmpMeshVertices.positions;
    auto& normals = tmpMeshVertices.normals;
    auto& uvs = tmpMeshVertices.uvs;

    size_t vertexCount = positions.size();
    size_t indexCount = tmpIndices.empty() ? vertexCount : tmpIndices.size();

    // TODO: other attributes

    Type_STLVector<meshopt_Stream> streams = {
        {positions.data(), sizeof(positions[0]), sizeof(positions[0])},
        {normals.data(), sizeof(normals[0]), sizeof(normals[0])},
        {uvs.data(), sizeof(uvs[0]), sizeof(uvs[0])}};

    Type_STLVector<uint32_t> remap(indexCount);
    vertexCount = meshopt_generateVertexRemapMulti(
        remap.data(), tmpIndices.empty() ? nullptr : tmpIndices.data(),
        indexCount, vertexCount, streams.data(), streams.size());

    ::std::decay_t<decltype(positions)> optimizedVertexPositions(vertexCount);
    ::std::decay_t<decltype(normals)> optimizedVertexNormals(vertexCount);
    ::std::decay_t<decltype(uvs)> optimizedVertexUVs(vertexCount);

    Type_STLVector<uint32_t> optimizedIndices(indexCount);

    meshopt_remapVertexBuffer(optimizedVertexPositions.data(), positions.data(),
                              indexCount, sizeof(positions[0]), remap.data());

    meshopt_remapVertexBuffer(optimizedVertexNormals.data(), normals.data(),
                              indexCount, sizeof(normals[0]), remap.data());

    meshopt_remapVertexBuffer(optimizedVertexUVs.data(), uvs.data(), indexCount,
                              sizeof(uvs[0]), remap.data());

    meshopt_remapIndexBuffer(optimizedIndices.data(), tmpIndices.data(),
                             indexCount, remap.data());

    positions = std::move(optimizedVertexPositions);
    normals = std::move(optimizedVertexNormals);
    uvs = std::move(optimizedVertexUVs);
    tmpIndices = std::move(optimizedIndices);
}

void RemapIndex(Type_STLVector<InternalMeshData>& tmpVertices,
                Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    if (tmpIndices.empty()) {
        tmpIndices.resize(tmpVertices.size());
    } else {
        assert(tmpIndices.size() == tmpVertices.size());
    }

    for (uint32_t i = 0; i < tmpVertices.size(); ++i) {
        RemapIndex(tmpVertices[i], tmpIndices[i]);
    }
}

void OptimizeMesh(InternalMeshData& tmpMeshVertices,
                  Type_STLVector<uint32_t>& tmpIndices) {
    auto& positions = tmpMeshVertices.positions;
    auto& normals = tmpMeshVertices.normals;
    auto& uvs = tmpMeshVertices.uvs;

    auto indexCount = tmpIndices.size();
    auto vertexCount = positions.size();

    const auto indices = tmpIndices.data();

    meshopt_optimizeVertexCache(indices, indices, indexCount, vertexCount);

    meshopt_optimizeOverdraw(indices, indices, indexCount,
                             reinterpret_cast<const float*>(positions.data()),
                             vertexCount, sizeof(positions[0]), 1.05f);

    Type_STLVector<uint32_t> remap(vertexCount);
    meshopt_optimizeVertexFetchRemap(remap.data(), indices, indexCount,
                                     vertexCount);

    meshopt_remapVertexBuffer(positions.data(), positions.data(), vertexCount,
                              sizeof(positions[0]), remap.data());

    meshopt_remapVertexBuffer(normals.data(), normals.data(), vertexCount,
                              sizeof(normals[0]), remap.data());

    meshopt_remapVertexBuffer(uvs.data(), uvs.data(), vertexCount,
                              sizeof(uvs[0]), remap.data());

    meshopt_remapIndexBuffer(indices, indices, indexCount, remap.data());
}

void OptimizeData(Type_STLVector<InternalMeshData>& tmpVertices,
                  Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    assert(tmpVertices.size() == tmpIndices.size());
    for (uint32_t i = 0; i < tmpVertices.size(); ++i) {
        OptimizeMesh(tmpVertices[i], tmpIndices[i]);
    }
}

void BuildMeshlet(InternalMeshlet& meshlet,
                  InternalMeshData const& tmpMeshVertices,
                  Type_STLVector<uint32_t> const& tmpIndices) {
    auto const& positions = tmpMeshVertices.positions;

    size_t maxMeshlets =
        meshopt_buildMeshletsBound(tmpIndices.size(), MESHLET_MAX_VERTEX_COUNT,
                                   MESHLET_MAX_TRIANGLE_COUNT);

    meshlet.infos.resize(maxMeshlets);
    meshlet.vertIndices.resize(maxMeshlets * MESHLET_MAX_VERTEX_COUNT);
    meshlet.triangles.resize(maxMeshlets * MESHLET_MAX_TRIANGLE_COUNT * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        (meshopt_Meshlet*)meshlet.infos.data(), meshlet.vertIndices.data(),
        meshlet.triangles.data(), tmpIndices.data(), tmpIndices.size(),
        (const float*)positions.data(), positions.size(), sizeof(positions[0]),
        MESHLET_MAX_VERTEX_COUNT, MESHLET_MAX_TRIANGLE_COUNT, 0.0f);

    const auto& last = meshlet.infos[meshletCount - 1];

    meshlet.vertIndices.resize(last.vertexOffset + last.vertexCount);
    meshlet.triangles.resize(last.triangleOffset
                             + ((last.triangleCount * 3 + 3) & ~3));
    meshlet.infos.resize(meshletCount);

    for (uint32_t i = 0; i < meshletCount; ++i) {
        meshopt_optimizeMeshlet(
            &meshlet.vertIndices[meshlet.infos[i].vertexOffset],
            &meshlet.triangles[meshlet.infos[i].triangleOffset],
            meshlet.infos[i].triangleCount, meshlet.infos[i].vertexCount);
    }
}

void BuildMeshletDatas(Type_STLVector<InternalMeshlet>& tmpMeshlets,
                       Type_STLVector<InternalMeshData>& tmpVertices,
                       Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    tmpMeshlets.resize(tmpVertices.size());
    for (uint32_t i = 0; i < tmpVertices.size(); ++i) {
        BuildMeshlet(tmpMeshlets[i], tmpVertices[i], tmpIndices[i]);
    }
}

void GenerateMeshletVertices(Type_STLVector<InternalMeshlet>& meshlets,
                             Type_STLVector<InternalMeshData>& tmpVertices) {
    auto meshCount = tmpVertices.size();
    Type_STLVector<InternalMeshData> outPos(meshCount);

    for (uint32_t i = 0; i < meshCount; ++i) {
        auto& tmpMeshVertices = tmpVertices[i];
        auto& outMeshVertices = outPos[i];

        auto& meshlet = meshlets[i];
        auto meshletCount = meshlet.infos.size();
        auto vertCount = meshlet.vertIndices.size();

        outMeshVertices.positions.resize(vertCount);
        outMeshVertices.normals.resize(vertCount);
        outMeshVertices.uvs.resize(vertCount);

        uint32_t index {0};
        for (uint32_t j = 0; j < meshletCount; ++j) {
            auto& info = meshlet.infos[j];
            for (uint32_t k = 0; k < info.vertexCount; ++k) {
                uint32_t vertIdx = meshlet.vertIndices[info.vertexOffset + k];

                outMeshVertices.positions[index] =
                    tmpMeshVertices.positions[vertIdx];
                outMeshVertices.normals[index] =
                    tmpMeshVertices.normals[vertIdx];
                outMeshVertices.uvs[index] = tmpMeshVertices.uvs[vertIdx];

                meshlet.vertIndices[info.vertexOffset + k] = index;

                index++;
            }
        }
    }

    tmpVertices = std::move(outPos);
}

void Generate_CISDIModel_Meshlets(
    CISDI_3DModel& data, Type_STLVector<InternalMeshlet>& tmpMeshlets) {
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        auto& mesh = data.meshes[i];
        auto& tmpMeshlet = tmpMeshlets[i];
        mesh.header.meshletCount = tmpMeshlet.infos.size();
        mesh.header.vertexCount = tmpMeshlet.vertIndices.size();
        mesh.header.meshletTriangleCount = tmpMeshlet.triangles.size();

        mesh.meshlets.infos = ::std::move(tmpMeshlet.infos);
        mesh.meshlets.triangles = ::std::move(tmpMeshlet.triangles);
    }
}

void Generate_CISDIModel_MeshletBoundingBoxes(
    CISDI_3DModel& data, Type_STLVector<InternalMeshData>& tmpVertices) {
    for (uint32_t i = 0; i < data.meshes.size(); ++i) {
        auto& mesh = data.meshes[i];
        auto& tmpPosVec = tmpVertices[i].positions;

        // meshlet bounding box
        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto& info = mesh.meshlets.infos[j];
            mesh.meshlets.boundingBoxes.resize(mesh.header.meshletCount);

            AABoundingBox bb {};

            for (uint32_t k = 0; k < info.vertexCount; ++k) {
                uint32_t vertIdx = info.vertexOffset + k;
                UpdateAABB(bb, tmpPosVec[vertIdx]);
            }
            mesh.meshlets.boundingBoxes[j] = bb;
        }

        // mesh bounding box
        for (auto const& meshletbb : mesh.meshlets.boundingBoxes) {
            UpdateAABB(mesh.boundingBox, meshletbb);
        }
    }

    // model bounding box
    for (auto const& meshbb : data.meshes) {
        UpdateAABB(data.boundingBox, meshbb.boundingBox);
    }
}

void Generate_CISDIModel_PackedVertices(
    CISDI_3DModel& data, Type_STLVector<InternalMeshData> const& tmpVertices) {
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        auto& mesh = data.meshes[i];
        auto const& tmpMeshVertices = tmpVertices[i];

        mesh.header.vertexCount = tmpMeshVertices.positions.size();

        mesh.meshlets.vertices.resize(mesh.header.meshletCount);

        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto vertCount = mesh.meshlets.infos[j].vertexCount;
            mesh.meshlets.vertices[j].positions.resize(vertCount);
            mesh.meshlets.vertices[j].normals.resize(vertCount);
            mesh.meshlets.vertices[j].uvs.resize(vertCount);
        }

        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto& info = mesh.meshlets.infos[j];
            auto& vertices = mesh.meshlets.vertices[j];
            for (uint32_t k = 0; k < info.vertexCount; ++k) {
                uint32_t vertIdx = info.vertexOffset + k;

                // position
                {
                    auto const& bb = mesh.meshlets.boundingBoxes[j];

                    Float32_3 fPos = tmpMeshVertices.positions[vertIdx];
                    Float32_3 bbLength {bb.max.x - bb.min.x,
                                        bb.max.y - bb.min.y,
                                        bb.max.z - bb.min.z};
                    Float32_3 encodedPos {
                        bbLength.x > 0.0f ? (fPos.x - bb.min.x) / bbLength.x
                                          : 0.0f,
                        bbLength.y > 0.0f ? (fPos.y - bb.min.y) / bbLength.y
                                          : 0.0f,
                        bbLength.z > 0.0f ? (fPos.z - bb.min.z) / bbLength.z
                                          : 0.0f};
                    UInt16_3 ui16Pos = {PackUnorm16(encodedPos.x),
                                        PackUnorm16(encodedPos.y),
                                        PackUnorm16(encodedPos.z)};
                    vertices.positions[k] = ui16Pos;
                }

                // normals
                {
                    Float32_3 fNorm = tmpMeshVertices.normals[vertIdx];

                    Float32_2 octNorm = UnitVectorToOctahedron(fNorm);

                    Int16_2 i16Norm = {PackSnorm16(octNorm.x),
                                       PackSnorm16(octNorm.y)};
                    vertices.normals[k] = i16Norm;
                }

                // uvs
                {
                    Float32_2 fUV = tmpMeshVertices.uvs[vertIdx];

                    // TODO: define uv wrap mode, using repeat for now
                    fUV = RepeatTexCoords(fUV);

                    UInt16_2 ui16UV = {PackUnorm16(fUV.x), PackUnorm16(fUV.y)};
                    vertices.uvs[k] = ui16UV;
                }
            }
        }
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

void WriteBoundingBox(std::ofstream& ofs, AABoundingBox const& box) {
    ofs.write((char*)&box, sizeof(box));
}

void WriteMeshes(std::ofstream& ofs,
                 Type_STLVector<CISDI_3DModel::Mesh> const& meshes) {
    for (auto const& mesh : meshes) {
        WriteMeshHeader(ofs, mesh.header);

        // TODO: other attributes

        // meshlet datas
        if (mesh.header.meshletCount > 0) {
            ofs.write(
                (char*)mesh.meshlets.infos.data(),
                sizeof(mesh.meshlets.infos[0]) * mesh.header.meshletCount);
        }

        if (mesh.header.meshletTriangleCount > 0) {
            ofs.write((char*)mesh.meshlets.triangles.data(),
                      sizeof(mesh.meshlets.triangles[0])
                          * mesh.header.meshletTriangleCount);
        }

        for (uint32_t i = 0; i < mesh.header.meshletCount; ++i) {
            auto const& vertices = mesh.meshlets.vertices[i];
            auto vertCount = mesh.meshlets.infos[i].vertexCount;

            ofs.write((char*)vertices.positions.data(),
                      sizeof(vertices.positions[0]) * vertCount);

            ofs.write((char*)vertices.normals.data(),
                      sizeof(vertices.normals[0]) * vertCount);

            ofs.write((char*)vertices.uvs.data(),
                      sizeof(vertices.uvs[0]) * vertCount);
        }

        if (mesh.header.meshletCount > 0) {
            ofs.write((char*)mesh.meshlets.boundingBoxes.data(),
                      sizeof(mesh.meshlets.boundingBoxes[0])
                          * mesh.header.meshletCount);
        }

        // bounding box
        WriteBoundingBox(ofs, mesh.boundingBox);
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
    WriteBoundingBox(out, data.boundingBox);
}

CISDI_3DModel Convert(const char* path, bool flipYZ, const char* output) {
    auto outputPath = ProcessOutputPath(path, output);

    CISDI_3DModel data {};

    // process input model file
    {
        Type_STLVector<InternalMeshData> tmpVertices {};
        Type_STLVector<Type_STLVector<uint32_t>> tmpIndices {};
        Type_STLVector<InternalMeshlet> tmpMeshlets {};

        if (::std::filesystem::path(path).extension() == ".fbx"
            || ::std::filesystem::path(path).extension() == ".FBX") {
            data = IntelliDesign_NS::ModelImporter::FBXSDK::Convert(
                path, flipYZ, tmpVertices, tmpIndices);
            RemapIndex(tmpVertices, tmpIndices);
        } else {
            data = IntelliDesign_NS::ModelImporter::Assimp::Convert(
                path, flipYZ, tmpVertices, tmpIndices);
        }

        OptimizeData(tmpVertices, tmpIndices);

        BuildMeshletDatas(tmpMeshlets, tmpVertices, tmpIndices);

        GenerateMeshletVertices(tmpMeshlets, tmpVertices);

        Generate_CISDIModel_Meshlets(data, tmpMeshlets);

        Generate_CISDIModel_MeshletBoundingBoxes(data, tmpVertices);

        Generate_CISDIModel_PackedVertices(data, tmpVertices);
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

        // TODO: other attributes

        mesh.meshlets.infos.resize(mesh.header.meshletCount);
        in.read((char*)mesh.meshlets.infos.data(),
                sizeof(mesh.meshlets.infos[0]) * mesh.header.meshletCount);

        mesh.meshlets.triangles.resize(mesh.header.meshletTriangleCount);
        in.read((char*)mesh.meshlets.triangles.data(),
                sizeof(mesh.meshlets.triangles[0])
                    * mesh.header.meshletTriangleCount);

        mesh.meshlets.vertices.resize(mesh.header.meshletCount);
        for (uint32_t i = 0; i < mesh.header.meshletCount; ++i) {
            auto& vertices = mesh.meshlets.vertices[i];
            auto vertCount = mesh.meshlets.infos[i].vertexCount;

            vertices.positions.resize(vertCount);
            in.read((char*)vertices.positions.data(),
                    sizeof(vertices.positions[0]) * vertCount);

            vertices.normals.resize(vertCount);
            in.read((char*)vertices.normals.data(),
                    sizeof(vertices.normals[0]) * vertCount);

            vertices.uvs.resize(vertCount);
            in.read((char*)vertices.uvs.data(),
                    sizeof(vertices.uvs[0]) * vertCount);
        }

        mesh.meshlets.boundingBoxes.resize(mesh.header.meshletCount);
        in.read(
            (char*)mesh.meshlets.boundingBoxes.data(),
            sizeof(mesh.meshlets.boundingBoxes[0]) * mesh.header.meshletCount);

        in.read((char*)&mesh.boundingBox, sizeof(mesh.boundingBox));
    }

    // read material
    data.materials.resize(data.header.materialCount);
    for (auto& mat : data.materials) {
        ReadString(in, mat.name);
        in.read((char*)&mat.ambient,
                sizeof(mat.ambient) * 3 + sizeof(mat.opacity));
    }

    in.read((char*)&data.boundingBox, sizeof(data.boundingBox));

    return data;
}

}  // namespace IntelliDesign_NS::ModelData