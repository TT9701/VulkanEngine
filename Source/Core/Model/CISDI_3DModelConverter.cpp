#include "CISDI_3DModelConverter.hpp"

#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include "Core/Utilities/Logger.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

CISDI_3DModelDataConverter::CISDI_3DModelDataConverter(
    const char* path, const char* outputDirectory, bool flipYZ)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(Utils::GetDirectory(path)),
      mOutputDirectory(GetOutputDirectory(outputDirectory)),
      mName(Utils::GetFileName(path)) {}

std::string CISDI_3DModelDataConverter::Execute() {
    auto name            = Utils::GetFileName(mPath);
    auto cisdiBinaryPath = mOutputDirectory + name + CISDI_3DModel_Subfix;

    Assimp::Importer importer {};

    const auto scene =
        importer.ReadFile(mPath, aiProcessPreset_TargetRealtime_Fast);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode) {
        // TODO: Logging
        DBG_LOG_INFO(::std::string("ERROR::ASSIMP::")
                     + importer.GetErrorString());
        return "";
    }

    ::std::ofstream cisdiBinary(cisdiBinaryPath,
                                ::std::ios::out | ::std::ios::binary);

    if (!cisdiBinary.is_open()) {
        throw ::std::runtime_error("fail to open file: " + cisdiBinaryPath);
    }

    WriteDataHeader(cisdiBinary,
                    {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                     CalcMeshCount(scene->mRootNode)});

    ProcessNode(cisdiBinary, scene->mRootNode, scene);

    return cisdiBinaryPath;
}

std::vector<Mesh> CISDI_3DModelDataConverter::LoadCISDIModelData(
    std::string const& path) {
    ::std::ifstream cisdiBinary(path, ::std::ios::binary);

    CISDI_3DModelData data {};
    cisdiBinary.read((char*)&data, sizeof(data.header));

    if (CISDI_3DModel_HEADER_UINT64 != data.header.header) {
        throw ::std::runtime_error(
            "Error::Cisdi3DModelConverter::LoadCISDIModelData " + path);
    }

    // TODO: Version Check

    ::std::vector<Mesh> meshes;
    meshes.reserve(data.header.meshCount);

    data.meshes.resize(data.header.meshCount);
    for (uint32_t i = 0; i < data.header.meshCount; ++i) {
        cisdiBinary.read((char*)&data.meshes[i], 8);

        ::std::vector<Vertex> vertices;
        vertices.reserve(data.meshes[i].header.vertexCount);
        for (uint32_t j = 0; j < data.meshes[i].header.vertexCount; ++j) {
            Vertex v {};
            cisdiBinary.read(
                (char*)&v.position,
                sizeof(CISDI_3DModelData::CISDI_Mesh::Vertex::position));
            cisdiBinary.read(
                (char*)&v.normal,
                sizeof(CISDI_3DModelData::CISDI_Mesh::Vertex::normal));
            vertices.push_back(v);
        }

        ::std::vector<uint32_t> indices;
        indices.resize(data.meshes[i].header.indexCount);
        cisdiBinary.read((char*)indices.data(),
                         sizeof(uint32_t) * data.meshes[i].header.indexCount);

        meshes.emplace_back(vertices, indices);
    }

    return meshes;
}

std::string CISDI_3DModelDataConverter::GetOutputDirectory(
    const std::string& output) {
    auto str = output.empty() ? Utils::GetDirectory(mDirectory) : output;
    if (!str.ends_with("/"))
        str += "/";
    return str;
}

void CISDI_3DModelDataConverter::ProcessNode(::std::ofstream& out, aiNode* node,
                                             const aiScene* scene) {
    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(out, mesh);
    }
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(out, node->mChildren[i], scene);
    }
}

void CISDI_3DModelDataConverter::ProcessMesh(::std::ofstream& out,
                                             aiMesh*          mesh) {
    WriteMeshHeader(out, {mesh->mNumVertices, mesh->mNumFaces * 3});

    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        glm::vec3 temp;

        // position
        temp.x = mesh->mVertices[i].x;
        if (mFlipYZ) {
            temp.y = mesh->mVertices[i].z;
            temp.z = mesh->mVertices[i].y;
            out.write((char*)&temp, sizeof(temp));
        } else {
            temp.y = mesh->mVertices[i].y;
            temp.z = mesh->mVertices[i].z;
            out.write((char*)&temp, sizeof(temp));
        }

        // normal
        temp.x = mesh->mNormals[i].x;
        if (mFlipYZ) {
            temp.y = mesh->mNormals[i].z;
            temp.z = mesh->mNormals[i].y;
            out.write((char*)&temp, sizeof(temp));
        } else {
            temp.y = mesh->mNormals[i].y;
            temp.z = mesh->mNormals[i].z;
            out.write((char*)&temp, sizeof(temp));
        }

        // // texcoords
        // if (mesh->HasTextureCoords(i)) {
        //     glm::vec2 vec2;
        //     vec2.x           = mesh->mTextureCoords[0][i].x;
        //     vec2.y           = mesh->mTextureCoords[0][i].y;
        // }
        //
        // // tangents and bitangents
        // if (mesh->HasTangentsAndBitangents()) {
        //     temp.x = mesh->mTangents[i].x;
        //     if (mFlipYZ) {
        //         temp.y = mesh->mTangents[i].z;
        //         temp.z = mesh->mTangents[i].y;
        //     } else {
        //         temp.y = mesh->mTangents[i].y;
        //         temp.z = mesh->mTangents[i].z;
        //     }
        //
        //     temp.x = mesh->mBitangents[i].x;
        //     if (mFlipYZ) {
        //         temp.y = mesh->mBitangents[i].z;
        //         temp.z = mesh->mBitangents[i].y;
        //     } else {
        //         temp.y = mesh->mBitangents[i].y;
        //         temp.z = mesh->mBitangents[i].z;
        //     }
        // }
    }

    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        auto face = mesh->mFaces[i];
        out.write((char*)face.mIndices, sizeof(face.mIndices[0]) * 3);
    }
}

uint32_t CISDI_3DModelDataConverter::CalcMeshCount(aiNode* node) {
    uint32_t meshCount {node->mNumMeshes};
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        meshCount += CalcMeshCount(node->mChildren[i]);
    }
    return meshCount;
}

void CISDI_3DModelDataConverter::WriteDataHeader(
    std::ofstream& ofs, CISDI_3DModelData::Header header) {
    ofs.write((char*)&header, sizeof(header));
}

void CISDI_3DModelDataConverter::WriteMeshHeader(
    std::ofstream& ofs, CISDI_3DModelData::CISDI_Mesh::MeshHeader meshHeader) {
    ofs.write((char*)&meshHeader, sizeof(meshHeader));
}