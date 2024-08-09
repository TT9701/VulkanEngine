#include "CISDI_3DModelData.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

namespace {

uint32_t CalcMeshCount(aiNode* node) {
    uint32_t meshCount {node->mNumMeshes};
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        meshCount += CalcMeshCount(node->mChildren[i]);
    }
    return meshCount;
}

void WriteDataHeader(std::ofstream& ofs, CISDI_3DModelData::Header header) {
    ofs.write((char*)&header, sizeof(header));
}

void WriteMeshHeader(std::ofstream&                            ofs,
                     CISDI_3DModelData::CISDI_Mesh::MeshHeader meshHeader) {
    ofs.write((char*)&meshHeader, sizeof(meshHeader));
}

void ProcessMesh(std::ofstream& out, aiMesh* mesh, bool flipYZ) {
    WriteMeshHeader(out, {mesh->mNumVertices, mesh->mNumFaces * 3});

    // position
    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        float temp[3];

        temp[0] = mesh->mVertices[i].x;
        if (flipYZ) {
            temp[1] = mesh->mVertices[i].z;
            temp[2] = mesh->mVertices[i].y;
        } else {
            temp[1] = mesh->mVertices[i].y;
            temp[2] = mesh->mVertices[i].z;
        }
        out.write((char*)&temp, sizeof(temp));
    }

    // normal
    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        float temp[3];
        temp[0] = mesh->mNormals[i].x;
        if (flipYZ) {
            temp[1] = mesh->mNormals[i].z;
            temp[2] = mesh->mNormals[i].y;
        } else {
            temp[1] = mesh->mNormals[i].y;
            temp[2] = mesh->mNormals[i].z;
        }
        out.write((char*)&temp, sizeof(temp));
    }

    // // texcoords
    // for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
    //     float temp[2];
    //     temp[0] = mesh->mTextureCoords[0][i].x;
    //     temp[1] = mesh->mTextureCoords[0][i].y;
    //     out.write((char*)&temp, sizeof(temp));
    // }
    //
    // // tangent
    // for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
    //     float temp[3];
    //     temp[0] = mesh->mTangents[i].x;
    //     if (flipYZ) {
    //         temp[1] = mesh->mTangents[i].z;
    //         temp[2] = mesh->mTangents[i].y;
    //     } else {
    //         temp[1] = mesh->mTangents[i].y;
    //         temp[2] = mesh->mTangents[i].z;
    //     }
    //     out.write((char*)&temp, sizeof(temp));
    // }
    //
    // // bitangent
    // for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
    //     float temp[3];
    //     temp[0] = mesh->mBitangents[i].x;
    //     if (flipYZ) {
    //         temp[1] = mesh->mBitangents[i].z;
    //         temp[2] = mesh->mBitangents[i].y;
    //     } else {
    //         temp[1] = mesh->mBitangents[i].y;
    //         temp[2] = mesh->mBitangents[i].z;
    //     }
    //     out.write((char*)&temp, sizeof(temp));
    // }

    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        auto face = mesh->mFaces[i];
        out.write((char*)face.mIndices, sizeof(face.mIndices[0]) * 3);
    }
}

void ProcessNode(std::ofstream& out, aiNode* node, const aiScene* scene,
                 bool flipYZ) {
    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(out, mesh, flipYZ);
    }
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(out, node->mChildren[i], scene, flipYZ);
    }
}

}  // namespace

void CISDI_3DModelData::Convert(const char* path, bool flipYZ,
                                const char* output) {
    auto inPath     = ::std::filesystem::path(path);
    auto outputPath = ::std::filesystem::canonical(output);

    if (!output) {
        outputPath = inPath.replace_extension(CISDI_3DModel_Subfix);
    } else {
        if (::std::filesystem::is_directory(outputPath)) {
            outputPath = outputPath.wstring().append(L"/").append(
                inPath.filename().replace_extension(CISDI_3DModel_Subfix));
        } else {
            throw ::std::runtime_error(
                "ERROR::CISDI_3DMODELDATA::CONVERT: Output is not a "
                "directory!");
        }
    }

    Assimp::Importer importer {};

    const auto scene =
        importer.ReadFile(path, aiProcessPreset_TargetRealtime_Fast);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode) {
        // TODO: Logging
        throw ::std::runtime_error(
            ::std::string("ERROR::CISDI_3DMODELDATA::CONVERT::ASSIMP: ")
            + importer.GetErrorString());
    }

    ::std::ofstream out(outputPath, ::std::ios::out | ::std::ios::binary);

    if (!out.is_open()) {
        throw ::std::runtime_error(::std::string("fail to open file: ")
                                   + outputPath.string());
    }

    WriteDataHeader(out, {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                          CalcMeshCount(scene->mRootNode)});

    ProcessNode(out, scene->mRootNode, scene, flipYZ);
}

CISDI_3DModelData CISDI_3DModelData::Load(const char* path) {
    ::std::ifstream in(path, ::std::ios::binary);
    if (!in.is_open()) {
        throw ::std::runtime_error(::std::string("fail to open file: ") + path);
    }

    CISDI_3DModelData data {};

    // Header check
    in.read((char*)&data, sizeof(data.header));
    if (CISDI_3DModel_HEADER_UINT64 != data.header.header) {
        throw ::std::runtime_error(
            ::std::string("Error::Cisdi3DModelConverter::LoadCISDIModelData ")
            + path);
    }

    // TODO: Version Check

    data.meshes.resize(data.header.meshCount);
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        in.read((char*)&data.meshes[i].header, sizeof(CISDI_Mesh::MeshHeader));

        data.meshes[i].vertices.positions.resize(
            data.meshes[i].header.vertexCount);
        in.read((char*)data.meshes[i].vertices.positions.data(),
                data.meshes[i].header.vertexCount
                    * sizeof(data.meshes[i].vertices.positions[0]));

        data.meshes[i].vertices.normals.resize(
            data.meshes[i].header.vertexCount);
        in.read((char*)data.meshes[i].vertices.normals.data(),
                data.meshes[i].header.vertexCount
                    * sizeof(data.meshes[i].vertices.normals[0]));
        // TODO: other attributes

        data.meshes[i].indices.resize(data.meshes[i].header.indexCount);
        in.read((char*)data.meshes[i].indices.data(),
                sizeof(uint32_t) * data.meshes[i].header.indexCount);
    }

    return data;
}