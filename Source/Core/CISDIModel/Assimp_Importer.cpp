#include "Assimp_Importer.h"

#include <iostream>
#include <stdexcept>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "CISDI_3DModelData.h"
#include "Common.h"

using namespace IntelliDesign_NS::ModelData;

namespace IntelliDesign_NS::ModelImporter {
namespace Assimp {

namespace {

uint32_t CalcMeshCount(aiNode* node) {
    uint32_t meshCount {node->mNumMeshes};
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        meshCount += CalcMeshCount(node->mChildren[i]);
    }
    return meshCount;
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

void ProcessMesh(CISDI_3DModel& data, aiMesh* mesh, bool flipYZ) {
    uint32_t vertCount = mesh->mNumVertices;

    CISDI_3DModel::Mesh cisdiMesh {};
    cisdiMesh.vertices.positions.resize(vertCount);
    cisdiMesh.vertices.normals.resize(vertCount);
    cisdiMesh.vertices.uvs.resize(vertCount);

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
    ReadNodeProperties(node);
    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(data, mesh, flipYZ);
    }
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(data, node->mChildren[i], scene, flipYZ);
    }

    auto material = *scene->mMaterials;
    aiColor3D color;
    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    aiString name;
    material->Get(AI_MATKEY_NAME, name);
    material->Get(AI_MATKEY_COLOR_AMBIENT, color);
    material->Get(AI_MATKEY_BASE_COLOR, color);

    for (uint32_t i = 0; i < material->mNumProperties; ++i) {
        auto& prop = material->mProperties[i];
        aiString key = prop->mKey;
    }
}

}  // namespace

CISDI_3DModel Convert(const char* path, bool flipYZ) {
    CISDI_3DModel data {};

    ::Assimp::Importer importer {};

    const auto scene = importer.ReadFile(
        path, aiProcessPreset_TargetRealtime_Fast/* | aiProcess_OptimizeMeshes*/
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
                   CalcMeshCount(scene->mRootNode), false};

    data.meshes.reserve(data.header.meshCount);

    ProcessNode(data, scene->mRootNode, scene, flipYZ);

    return data;
}

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
