#include "Assimp_Importer.h"

#include <iostream>
#include <stdexcept>

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <filesystem>
#include <map>

#include "CISDI_3DModelData.h"
#include "Common.h"

using namespace IntelliDesign_NS::ModelData;

namespace IntelliDesign_NS::ModelImporter {
namespace Assimp {

namespace {

uint32_t CalcNodeCount(aiNode* node) {
    uint32_t nodeCount {1};
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        nodeCount += CalcNodeCount(node->mChildren[i]);
    }
    return nodeCount;
}

void ReadNodeProperties(aiNode* node) {
    ::std::cout << node->mName.C_Str() << ::std::endl;
    if (node->mMetaData) {
        for (unsigned int i = 0; i < node->mMetaData->mNumProperties; ++i) {
            aiString key = node->mMetaData->mKeys[i];
            aiMetadataEntry entry = node->mMetaData->mValues[i];

            std::string keyStr(key.C_Str());

            switch (entry.mType) {
                case AI_BOOL:
                    std::cout << "AI_BOOL: " << keyStr << ": "
                              << (entry.mData ? "true" : "false") << std::endl;
                    break;
                case AI_INT32:
                    std::cout << "AI_INT32: " << keyStr << ": "
                              << *static_cast<int32_t*>(entry.mData)
                              << std::endl;
                    break;
                case AI_UINT64:
                    std::cout << "AI_UINT64: " << keyStr << ": "
                              << *static_cast<uint64_t*>(entry.mData)
                              << std::endl;
                    break;
                case AI_FLOAT:
                    std::cout << "AI_FLOAT: " << keyStr << ": "
                              << *static_cast<float*>(entry.mData) << std::endl;
                    break;
                case AI_DOUBLE:
                    std::cout << "AI_DOUBLE: " << keyStr << ": "
                              << *static_cast<double*>(entry.mData)
                              << std::endl;
                    break;
                case AI_AISTRING:
                    std::cout << "AI_AISTRING: " << keyStr << ": "
                              << static_cast<aiString*>(entry.mData)->C_Str()
                              << std::endl;
                    break;
                case AI_AIVECTOR3D: {
                    aiVector3D* vec = static_cast<aiVector3D*>(entry.mData);
                    std::cout << "AI_AIVECTOR3D: " << keyStr << ": (" << vec->x
                              << ", " << vec->y << ", " << vec->z << ")"
                              << std::endl;
                } break;
                default:
                    std::cout << keyStr << ": (unknown type)" << std::endl;
                    break;
            }
        }
    }
}

void ProcessMesh(CISDI_3DModel& data, CISDI_3DModel::Node& cisdiNode,
                 aiMesh* mesh, bool flipYZ,
                 Type_STLVector<Type_STLVector<Float32_3>>& tmpPos,
                 Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    uint32_t meshIdx = (uint32_t)data.meshes.size();
    uint32_t vertCount = mesh->mNumVertices;

    CISDI_3DModel::Mesh cisdiMesh {};
    cisdiMesh.vertices.normals.resize(vertCount);
    cisdiMesh.vertices.uvs.resize(vertCount);

    auto& tmpPosVec = tmpPos.emplace_back();
    tmpPosVec.resize(vertCount);

    // position
    for (uint32_t i = 0; i < vertCount; ++i) {
        Float32_3 temp {mesh->mVertices[i].x, mesh->mVertices[i].y,
                        mesh->mVertices[i].z};

        if (flipYZ)
            ::std::swap(temp[1], temp[2]);

        UpdateAABB(cisdiMesh.boundingBox, temp);

        tmpPosVec[i] = temp;
    }

    // normal
    for (uint32_t i = 0; i < vertCount; ++i) {
        Float32_3 normal = {mesh->mNormals[i].x, mesh->mNormals[i].y,
                            mesh->mNormals[i].z};

        if (flipYZ)
            ::std::swap(normal[1], normal[2]);

        Float32_2 octNorm {UnitVectorToOctahedron(
            {mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z})};

        // normal.x is runtime decoded in shader

        cisdiMesh.vertices.normals[i] =
            Int16_2 {PackSnorm16(octNorm.x), PackSnorm16(octNorm.y)};
    }

    // texcoords
    if (mesh->HasTextureCoords(0)) {
        for (uint32_t i = 0; i < vertCount; ++i) {
            Float32_2 temp {mesh->mTextureCoords[0][i].x,
                            mesh->mTextureCoords[0][i].y};

            // TODO: define uv wrap mode, using repeat for now
            temp = RepeatTexCoords(temp);

            cisdiMesh.vertices.uvs[i] =
                UInt16_2 {PackUnorm16(temp.x), PackUnorm16(temp.y)};
        }
    }

    // index
    uint32_t indexCount = mesh->mNumFaces * 3;

    Type_STLVector<uint32_t> indices {};
    indices.reserve(indexCount);

    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        auto face = mesh->mFaces[i];
        for (uint32_t j = 0; j < face.mNumIndices; ++j) {
            indices.push_back(*(face.mIndices + j));
        }
    }

    data.meshes.emplace_back(cisdiMesh);
    tmpIndices.emplace_back(::std::move(indices));

    cisdiNode.meshIdx = meshIdx;
    cisdiNode.materialIdx = mesh->mMaterialIndex;
}

uint32_t ProcessNode(CISDI_3DModel& data, uint32_t parentNodeIdx, aiNode* node,
                     const aiScene* scene, bool flipYZ,
                     Type_STLVector<Type_STLVector<Float32_3>>& tmpPos,
                     Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    // ReadNodeProperties(node);

    uint32_t nodeIdx = data.nodes.size();
    int childCount = node->mNumChildren;

    CISDI_3DModel::Node cisdiNode {};
    cisdiNode.name = node->mName.C_Str();
    cisdiNode.parentIdx = parentNodeIdx;
    cisdiNode.childCount = childCount;
    cisdiNode.childrenIdx.reserve(childCount);

    // node contains <= 1 mesh
    assert(node->mNumMeshes <= 1);

    if (node->mNumMeshes > 0)
        ProcessMesh(data, cisdiNode, scene->mMeshes[node->mMeshes[0]], flipYZ,
                    tmpPos, tmpIndices);

    auto& ref = data.nodes.emplace_back(::std::move(cisdiNode));

    for (uint32_t i = 0; i < childCount; ++i) {
        ref.childrenIdx.emplace_back(ProcessNode(data, nodeIdx,
                                                 node->mChildren[i], scene,
                                                 flipYZ, tmpPos, tmpIndices));
    }

    return nodeIdx;
}

}  // namespace

CISDI_3DModel Convert(const char* path, bool flipYZ,
                      Type_STLVector<Type_STLVector<Float32_3>>& tmpPos,
                      Type_STLVector<Type_STLVector<uint32_t>>& outIndices) {
    CISDI_3DModel data {};

    ::Assimp::Importer importer {};

    const auto scene =
        importer.ReadFile(path, aiProcessPreset_TargetRealtime_Fast
                                    | aiProcess_FixInfacingNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode) {
        // TODO: Logging
        throw ::std::runtime_error(
            (::std::string("ERROR::CISDI_3DModel::Convert::ASSIMP: ")
             + importer.GetErrorString())
                .c_str());
    }

    data.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                   CalcNodeCount(scene->mRootNode), scene->mNumMeshes,
                   scene->mNumMaterials};

    data.name = ::std::filesystem::path(path).stem().string();

    data.nodes.reserve(data.header.nodeCount);

    data.meshes.reserve(data.header.meshCount);
    tmpPos.reserve(data.header.meshCount);
    outIndices.reserve(data.header.meshCount);

    data.materials.reserve(data.header.materialCount);

    for (uint32_t i = 0; i < data.header.materialCount; ++i) {
        auto material = scene->mMaterials[i];
        CISDI_3DModel::Material cisdiMaterial {};
        material->Get(AI_MATKEY_NAME, cisdiMaterial.name);
        aiColor3D color;
        float opacity;
        material->Get(AI_MATKEY_COLOR_AMBIENT, color);
        cisdiMaterial.ambient = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        cisdiMaterial.diffuse = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
        cisdiMaterial.emissive = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_OPACITY, opacity);
        cisdiMaterial.opacity = opacity;
        data.materials.emplace_back(cisdiMaterial);
    }

    ProcessNode(data, ~0ui32, scene->mRootNode, scene, flipYZ, tmpPos,
                outIndices);

    for (auto const& mesh : data.meshes) {
        UpdateAABB(data.boundingBox, mesh.boundingBox);
    }

    // shrink to fit
    data.header.nodeCount = data.nodes.size();
    data.header.meshCount = data.meshes.size();
    data.header.materialCount = data.materials.size();

    return data;
}

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
