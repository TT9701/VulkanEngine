#include "Assimp_Importer.h"

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>

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

void ProcessNodeProperties(aiNode* node, CISDI_3DModel::Node& cisdiNode) {
    if (node->mMetaData) {
        for (unsigned int i = 0; i < node->mMetaData->mNumProperties; ++i) {
            aiString key = node->mMetaData->mKeys[i];
            aiMetadataEntry entry = node->mMetaData->mValues[i];

            std::string keyStr(key.C_Str());

            switch (entry.mType) {
                case AI_BOOL:
                    cisdiNode.userProperties[keyStr] = *(bool*)entry.mData;
                    break;
                case AI_INT32:
                    cisdiNode.userProperties[keyStr] = *(int32_t*)entry.mData;
                    break;
                case AI_UINT32:
                    cisdiNode.userProperties[keyStr] = *(uint32_t*)entry.mData;
                    break;
                case AI_INT64:
                    cisdiNode.userProperties[keyStr] = *(int64_t*)entry.mData;
                    break;
                case AI_UINT64:
                    cisdiNode.userProperties[keyStr] = *(uint64_t*)entry.mData;
                    break;
                case AI_FLOAT:
                    cisdiNode.userProperties[keyStr] = *(float*)entry.mData;
                    break;
                case AI_DOUBLE:
                    cisdiNode.userProperties[keyStr] = *(double*)entry.mData;
                    break;
                case AI_AISTRING:
                    cisdiNode.userProperties[keyStr] = std::string(
                        static_cast<aiString*>(entry.mData)->C_Str());
                    break;
                default:
                    std::cout << keyStr << ": (unknown type)" << std::endl;
                    break;
            }
        }
    }
}

void ProcessMesh(CISDI_3DModel::Node& cisdiNode, aiMesh* mesh, bool flipYZ,
                 Type_STLVector<InternalMeshData>& tmpVertices,
                 Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
    uint32_t vertCount = mesh->mNumVertices;

    uint32_t meshIdx = (uint32_t)tmpVertices.size();
    auto& tmpMeshVertices = tmpVertices.emplace_back();
    tmpMeshVertices.positions.resize(vertCount);
    tmpMeshVertices.normals.resize(vertCount);
    tmpMeshVertices.uvs.resize(vertCount);

    // position
    for (uint32_t i = 0; i < vertCount; ++i) {

        tmpMeshVertices.positions[i] = Float32_3 {
            mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};

        if (flipYZ)
            ::std::swap(tmpMeshVertices.positions[i][1],
                        tmpMeshVertices.positions[i][2]);
    }

    // normal
    for (uint32_t i = 0; i < vertCount; ++i) {
        tmpMeshVertices.normals[i] = Float32_3 {
            mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z};

        if (flipYZ)
            ::std::swap(tmpMeshVertices.normals[i][1],
                        tmpMeshVertices.normals[i][2]);
    }

    // texcoords
    if (mesh->HasTextureCoords(0)) {
        for (uint32_t i = 0; i < vertCount; ++i) {
            tmpMeshVertices.uvs[i] = Float32_2 {mesh->mTextureCoords[0][i].x,
                                                mesh->mTextureCoords[0][i].y};
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

    tmpIndices.emplace_back(::std::move(indices));

    cisdiNode.meshIdx = meshIdx;
    cisdiNode.materialIdx = mesh->mMaterialIndex;
}

uint32_t ProcessNode(CISDI_3DModel& data, uint32_t parentNodeIdx, aiNode* node,
                     const aiScene* scene, bool flipYZ,
                     Type_STLVector<InternalMeshData>& tmpVertices,
                     Type_STLVector<Type_STLVector<uint32_t>>& tmpIndices) {
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
        ProcessMesh(cisdiNode, scene->mMeshes[node->mMeshes[0]], flipYZ,
                    tmpVertices, tmpIndices);

    ProcessNodeProperties(node, cisdiNode);

    auto& ref = data.nodes.emplace_back(::std::move(cisdiNode));

    for (uint32_t i = 0; i < childCount; ++i) {
        ref.childrenIdx.emplace_back(
            ProcessNode(data, nodeIdx, node->mChildren[i], scene, flipYZ,
                        tmpVertices, tmpIndices));
    }

    return nodeIdx;
}

}  // namespace

CISDI_3DModel Convert(const char* path, bool flipYZ,
                      Type_STLVector<InternalMeshData>& tmpVertices,
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

    data.meshes.resize(data.header.meshCount);
    tmpVertices.reserve(data.header.meshCount);
    outIndices.reserve(data.header.meshCount);

    data.materials.reserve(data.header.materialCount);

    for (uint32_t i = 0; i < data.header.materialCount; ++i) {
        auto material = scene->mMaterials[i];
        Material cisdiMaterial {};
        material->Get(AI_MATKEY_NAME, cisdiMaterial.name);
        aiColor3D color;
        float opacity;
        material->Get(AI_MATKEY_COLOR_AMBIENT, color);
        cisdiMaterial.data.ambient = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        cisdiMaterial.data.diffuse = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
        cisdiMaterial.data.emissive = {color.r, color.g, color.b};
        material->Get(AI_MATKEY_OPACITY, opacity);
        cisdiMaterial.data.transparency = opacity;
        data.materials.emplace_back(cisdiMaterial);
    }

    ProcessNode(data, ~0ui32, scene->mRootNode, scene, flipYZ, tmpVertices,
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
