#include "CISDI_3DModelData.h"

#include <meshoptimizer.h>

#include "Source/Common/FileIO.h"
#include "Source/Common/Math.h"
#include "Source/Importer/Assimp_Importer.h"
#include "Source/Importer/Combined_Importer.h"
#include "Source/Importer/FBX_Importer.h"

namespace IntelliDesign_NS::ModelData {

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
    const auto pRemap = remap.data();

    vertexCount = meshopt_generateVertexRemapMulti(
        pRemap, tmpIndices.empty() ? nullptr : tmpIndices.data(), indexCount,
        vertexCount, streams.data(), streams.size());

    ::std::decay_t<decltype(positions)> optimizedVertexPositions(vertexCount);
    ::std::decay_t<decltype(normals)> optimizedVertexNormals(vertexCount);
    ::std::decay_t<decltype(uvs)> optimizedVertexUVs(vertexCount);

    Type_STLVector<uint32_t> optimizedIndices(indexCount);

    meshopt_remapVertexBuffer(optimizedVertexPositions.data(), positions.data(),
                              indexCount, sizeof(positions[0]), pRemap);

    meshopt_remapVertexBuffer(optimizedVertexNormals.data(), normals.data(),
                              indexCount, sizeof(normals[0]), pRemap);

    meshopt_remapVertexBuffer(optimizedVertexUVs.data(), uvs.data(), indexCount,
                              sizeof(uvs[0]), pRemap);

    meshopt_remapIndexBuffer(optimizedIndices.data(), tmpIndices.data(),
                             indexCount, pRemap);

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

    const auto pIndices = tmpIndices.data();

    meshopt_optimizeVertexCache(pIndices, pIndices, indexCount, vertexCount);

    meshopt_optimizeOverdraw(pIndices, pIndices, indexCount,
                             reinterpret_cast<const float*>(positions.data()),
                             vertexCount, sizeof(positions[0]), 1.05f);

    Type_STLVector<uint32_t> remap(vertexCount);
    const auto pRemap = remap.data();

    meshopt_optimizeVertexFetchRemap(pRemap, pIndices, indexCount, vertexCount);

    meshopt_remapVertexBuffer(positions.data(), positions.data(), vertexCount,
                              sizeof(positions[0]), pRemap);

    meshopt_remapVertexBuffer(normals.data(), normals.data(), vertexCount,
                              sizeof(normals[0]), pRemap);

    meshopt_remapVertexBuffer(uvs.data(), uvs.data(), vertexCount,
                              sizeof(uvs[0]), pRemap);

    meshopt_remapIndexBuffer(pIndices, pIndices, indexCount, pRemap);
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
        reinterpret_cast<meshopt_Meshlet*>(meshlet.infos.data()),
        meshlet.vertIndices.data(), meshlet.triangles.data(), tmpIndices.data(),
        tmpIndices.size(), reinterpret_cast<const float*>(positions.data()),
        positions.size(), sizeof(positions[0]), MESHLET_MAX_VERTEX_COUNT,
        MESHLET_MAX_TRIANGLE_COUNT, 0.0f);

    const auto& last = meshlet.infos[meshletCount - 1];

    meshlet.vertIndices.resize(last.vertexOffset + last.vertexCount);
    meshlet.triangles.resize(last.triangleOffset
                             + ((last.triangleCount * 3 + 3) & ~3));
    meshlet.infos.resize(meshletCount);

    for (uint32_t i = 0; i < meshletCount; ++i) {
        auto& info = meshlet.infos[i];
        meshopt_optimizeMeshlet(&meshlet.vertIndices[info.vertexOffset],
                                &meshlet.triangles[info.triangleOffset],
                                info.triangleCount, info.vertexCount);
    }
}

void BuildMeshletDatas(
    Type_STLVector<InternalMeshlet>& tmpMeshlets,
    Type_STLVector<InternalMeshData> const& tmpVertices,
    Type_STLVector<Type_STLVector<uint32_t>> const& tmpIndices,
    ::std::pmr::memory_resource* pMemPool) {
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

        auto& infos =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>();
        infos = ::std::move(tmpMeshlet.infos);

        auto& triangles =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Triangle>();
        triangles = ::std::move(tmpMeshlet.triangles);
    }
}

void Generate_CISDIModel_MeshletBoundingBoxes(
    CISDI_3DModel& data, Type_STLVector<InternalMeshData> const& tmpVertices,
    ::std::pmr::memory_resource* pMemPool) {
    using namespace Core::MathCore;

    for (uint32_t i = 0; i < data.meshes.size(); ++i) {
        auto& mesh = data.meshes[i];
        auto const& tmpPosVec = tmpVertices[i].positions;

        auto const& infos =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>();

        auto& boundingBoxes =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::BoundingBox>();
        boundingBoxes =
            Type_STLVector<AABoundingBox>(mesh.header.meshletCount, pMemPool);

        // meshlet bounding box
        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto const& info = infos[j];

            AABoundingBox bb {};

            for (uint32_t k = 0; k < info.vertexCount; ++k) {
                uint32_t vertIdx = info.vertexOffset + k;
                auto const& tmpPos = tmpPosVec[vertIdx];
                if (bb.Contains(SIMD_Vec {tmpPos.x, tmpPos.y, tmpPos.z})
                    == ContainmentType::DISJOINT) {
                    constexpr size_t pointCount = 9;
                    Float3 tmp[pointCount];
                    bb.GetCorners(tmp);
                    tmp[8] = {tmpPos.x, tmpPos.y, tmpPos.z};
                    AABoundingBox::CreateFromPoints(bb, pointCount, tmp,
                                                    sizeof(tmp[0]));
                }
            }
            boundingBoxes[j] = bb;
        }

        // mesh bounding box
        for (auto const& meshletbb : boundingBoxes) {
            AABoundingBox::CreateMerged(mesh.boundingBox, mesh.boundingBox,
                                        meshletbb);
        }
    }

    // model bounding box
    for (auto const& meshbb : data.meshes) {
        AABoundingBox::CreateMerged(data.boundingBox, data.boundingBox,
                                    meshbb.boundingBox);
    }
}

void Generate_CISDIModel_PackedVertices(
    CISDI_3DModel& data, Type_STLVector<InternalMeshData> const& tmpVertices,
    ::std::pmr::memory_resource* pMemPool) {
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        auto& mesh = data.meshes[i];
        auto const& tmpMeshVertices = tmpVertices[i];

        mesh.header.vertexCount = tmpMeshVertices.positions.size();

        auto& vertices =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Vertex>();

        vertices.resize(mesh.header.meshletCount);

        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto vertCount =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>()[j]
                    .vertexCount;

            vertices[j].GetProperty<VertexAttributeEnum::Position>() =
                Type_STLVector<UInt16_3>(vertCount, pMemPool);

            vertices[j].GetProperty<VertexAttributeEnum::Normal>() =
                Type_STLVector<Int16_2>(vertCount, pMemPool);

            vertices[j].GetProperty<VertexAttributeEnum::UV>() =
                Type_STLVector<UInt16_2>(vertCount, pMemPool);
        }

        for (uint32_t j = 0; j < mesh.header.meshletCount; ++j) {
            auto const& info =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>()[j];

            auto& meshletVertices =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Vertex>()[j];

            auto const& bb =
                mesh.meshlets
                    .GetProperty<MeshletPropertyTypeEnum::BoundingBox>()[j];

            for (uint32_t k = 0; k < info.vertexCount; ++k) {
                uint32_t vertIdx = info.vertexOffset + k;

                // position
                {
                    Float32_3 fPos = tmpMeshVertices.positions[vertIdx];

                    auto extent = bb.Extents;
                    Float32_3 bbLength {extent.x * 2.0f, extent.y * 2.0f,
                                        extent.z * 2.0f};

                    auto bbMinVec = Core::MathCore::VectorSubtract(
                        bb.Center.GetSIMD(), extent.GetSIMD());

                    Core::MathCore::Float3 bbMin {};
                    DirectX::XMStoreFloat3(&bbMin, bbMinVec);

                    Float32_3 encodedPos {
                        bbLength.x > 0.0f ? (fPos.x - bbMin.x) / bbLength.x
                                          : 0.0f,
                        bbLength.y > 0.0f ? (fPos.y - bbMin.y) / bbLength.y
                                          : 0.0f,
                        bbLength.z > 0.0f ? (fPos.z - bbMin.z) / bbLength.z
                                          : 0.0f};
                    UInt16_3 ui16Pos = {PackUnorm16(encodedPos.x),
                                        PackUnorm16(encodedPos.y),
                                        PackUnorm16(encodedPos.z)};
                    meshletVertices
                        .GetProperty<VertexAttributeEnum::Position>()[k] =
                        ui16Pos;
                }

                // normals
                {
                    Float32_3 fNorm = tmpMeshVertices.normals[vertIdx];

                    Float32_2 octNorm = UnitVectorToOctahedron(fNorm);

                    Int16_2 i16Norm = {PackSnorm16(octNorm.x),
                                       PackSnorm16(octNorm.y)};
                    meshletVertices
                        .GetProperty<VertexAttributeEnum::Normal>()[k] =
                        i16Norm;
                }

                // uvs
                {
                    Float32_2 fUV = tmpMeshVertices.uvs[vertIdx];

                    // TODO: define uv wrap mode, using repeat for now
                    fUV = RepeatTexCoords(fUV);

                    UInt16_2 ui16UV = {PackUnorm16(fUV.x), PackUnorm16(fUV.y)};
                    meshletVertices.GetProperty<VertexAttributeEnum::UV>()[k] =
                        ui16UV;
                }
            }
        }
    }
}

CISDI_3DModel::CISDI_3DModel(std::pmr::memory_resource* pMemPool)
    : name(pMemPool), nodes(pMemPool), meshes(pMemPool), materials(pMemPool) {}

CISDI_3DModel::TempObject::Type_PInstance Convert(
    CISDI_3DModel* model, const char* path, bool flipYZ,
    ::std::pmr::memory_resource* pMemPool, const char* output) {
    auto outputPath = ProcessOutputPath(path, output);

    auto& data = *model;

    CISDI_3DModel::TempObject::Type_PInstance pFBXImporter {nullptr};
    // process input model file
    {
        Type_STLVector<InternalMeshData> tmpVertices {pMemPool};
        Type_STLVector<Type_STLVector<uint32_t>> tmpIndices {pMemPool};
        Type_STLVector<InternalMeshlet> tmpMeshlets {pMemPool};

        if (::std::filesystem::path(path).extension() == ".fbx"
            || ::std::filesystem::path(path).extension() == ".FBX") {
            if (bUseCombinedImport) {
                IntelliDesign_NS::ModelImporter::CombinedImporter importer {
                    pMemPool, path, flipYZ, data, tmpVertices, tmpIndices};
                pFBXImporter = importer.ExtractFBXImporter();
            } else {
                pFBXImporter =
                    CMP_NS::New_Unique<ModelImporter::FBXSDK::Importer>(
                        pMemPool, pMemPool, path, flipYZ, data, tmpVertices,
                        tmpIndices);
                RemapIndex(tmpVertices, tmpIndices);
            }
        } else {
            IntelliDesign_NS::ModelImporter::Assimp::Importer importer {
                pMemPool, path, flipYZ, data, tmpVertices, tmpIndices};
        }

        OptimizeData(tmpVertices, tmpIndices);

        BuildMeshletDatas(tmpMeshlets, tmpVertices, tmpIndices, pMemPool);

        GenerateMeshletVertices(tmpMeshlets, tmpVertices);

        Generate_CISDIModel_Meshlets(data, tmpMeshlets);

        Generate_CISDIModel_MeshletBoundingBoxes(data, tmpVertices, pMemPool);

        Generate_CISDIModel_PackedVertices(data, tmpVertices, pMemPool);
    }

    Write_CISDI_File(outputPath.c_str(), data);

    return pFBXImporter;
}

void Load(CISDI_3DModel* model, const char* path,
          ::std::pmr::memory_resource* pMemPool) {
    Read_CISDI_File(model, path, pMemPool);
}

}  // namespace IntelliDesign_NS::ModelData