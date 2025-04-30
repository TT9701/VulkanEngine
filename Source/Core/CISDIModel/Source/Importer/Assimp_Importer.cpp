#include "Assimp_Importer.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <assimp/postprocess.h>

#include <Core/System/GameTimer.h>
#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

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

Type_STLString ToSTLString(aiString const& string) {
    return string.C_Str();
}

}  // namespace

Importer::Importer(std::pmr::memory_resource* pMemPool, const char* path,
                   bool flipYZ, CISDI_3DModel& outData,
                   Type_InternalMeshDatas& tmpVertices,
                   Type_Indices& outIndices)
    : pMemPool(pMemPool), mTmpVertices(tmpVertices), mOutIndices(outIndices) {
    ImportScene(path);
    InitializeData(outData, path);
    ExtractMaterials(outData);
    ProcessNode(outData, ~0ui32, mScene->mRootNode, flipYZ);

    outData.header.meshCount = (uint32_t)mTmpVertices.size();
    outData.meshes.resize(outData.header.meshCount);
}

Importer::~Importer() {}

void Importer::ImportScene(const char* path) {

    mScene = const_cast<aiScene*>(importer.ReadFile(
        path,
        aiProcessPreset_TargetRealtime_Fast | aiProcess_FixInfacingNormals));

    if (!mScene || mScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !mScene->mRootNode) {
        // TODO: Logging
        throw ::std::runtime_error(
            (::std::string("ERROR::CISDI_3DModel::Convert::ASSIMP: ")
             + importer.GetErrorString())
                .c_str());
    }
}

void Importer::InitializeData(CISDI_3DModel& outData, const char* path) {
    outData.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                      CalcNodeCount(mScene->mRootNode), mScene->mNumMeshes,
                      mScene->mNumMaterials};

    outData.name = ::std::filesystem::path(path).stem().string().c_str();

    outData.nodes.reserve(outData.header.nodeCount);

    mTmpVertices.reserve(outData.header.meshCount);
    mOutIndices.reserve(outData.header.meshCount);

    outData.materials.reserve(outData.header.materialCount);
}

void Importer::ExtractMaterials(CISDI_3DModel& outData) {
    for (uint32_t i = 0; i < mScene->mNumMaterials; ++i) {
        auto material = mScene->mMaterials[i];
        CISDI_Material cisdiMaterial {pMemPool};
        aiString name;
        material->Get(AI_MATKEY_NAME, name);
        cisdiMaterial.name = ToSTLString(name);
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
        outData.materials.emplace_back(cisdiMaterial);
    }
}

uint32_t Importer::ProcessNode(CISDI_3DModel& outData, uint32_t parentNodeIdx,
                               aiNode* node, bool flipYZ) {
    uint32_t nodeIdx = outData.nodes.size();
    int childCount = node->mNumChildren;

    CISDI_Node cisdiNode {pMemPool};
    cisdiNode.name = node->mName.C_Str();
    cisdiNode.parentIdx = parentNodeIdx;
    cisdiNode.childCount = childCount;
    cisdiNode.childrenIdx.reserve(childCount);

    // node contains <= 1 mesh
    assert(node->mNumMeshes <= 1);

    if (node->mNumMeshes > 0)
        ProcessMesh(cisdiNode, mScene->mMeshes[node->mMeshes[0]], flipYZ);

    ProcessNodeProperties(node, cisdiNode);

    auto& ref = outData.nodes.emplace_back(::std::move(cisdiNode));

    for (uint32_t i = 0; i < childCount; ++i) {
        ref.childrenIdx.emplace_back(
            ProcessNode(outData, nodeIdx, node->mChildren[i], flipYZ));
    }

    return nodeIdx;
}

void Importer::ProcessMesh(ModelData::CISDI_Node& cisdiNode, aiMesh* mesh,
                           bool flipYZ) {
    uint32_t vertCount = mesh->mNumVertices;

    if (vertCount == 0)
        return;

    uint32_t meshIdx = (uint32_t)mTmpVertices.size();
    auto& tmpMeshVertices = mTmpVertices.emplace_back();
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

    mOutIndices.emplace_back(::std::move(indices));

    cisdiNode.meshIdx = meshIdx;
    cisdiNode.materialIdx = mesh->mMaterialIndex;
}

void Importer::ProcessNodeProperties(aiNode* node, CISDI_Node& cisdiNode) {
    if (node->mMetaData) {
        for (unsigned int i = 0; i < node->mMetaData->mNumProperties; ++i) {
            const char* key = node->mMetaData->mKeys[i].C_Str();
            aiMetadataEntry entry = node->mMetaData->mValues[i];

            switch (entry.mType) {
                case AI_BOOL:
                    cisdiNode.userProperties[key] = *(bool*)entry.mData;
                    break;
                case AI_INT32:
                    cisdiNode.userProperties[key] = *(int32_t*)entry.mData;
                    break;
                case AI_UINT32:
                    cisdiNode.userProperties[key] = *(uint32_t*)entry.mData;
                    break;
                case AI_INT64:
                    cisdiNode.userProperties[key] = *(int64_t*)entry.mData;
                    break;
                case AI_UINT64:
                    cisdiNode.userProperties[key] = *(uint64_t*)entry.mData;
                    break;
                case AI_FLOAT:
                    cisdiNode.userProperties[key] = *(float*)entry.mData;
                    break;
                case AI_DOUBLE:
                    cisdiNode.userProperties[key] = *(double*)entry.mData;
                    break;
                case AI_AISTRING:
                    cisdiNode.userProperties[key] =
                        static_cast<aiString*>(entry.mData)->C_Str();
                    break;
                default:
                    // std::cout << key << ": (unknown type)" << std::endl;
                    break;
            }
        }

        if (node->mMetaData->mNumProperties > 0)
            cisdiNode.userProperties["_NumProperties_"] =
                node->mMetaData->mNumProperties;

        cisdiNode.userPropertyCount = cisdiNode.userProperties.size();
    }
}

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
