#include "Model.hpp"

#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
// #include <assimp/DefaultLogger.hpp>

#include "Core/Utilities/Logger.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

Model::Model(std::string const& path, bool flipYZ)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(Utils::GetDirectory(path)),
      mName(Utils::GetFileName(path)) {
    LoadModel();
}

Model::Model(std::vector<Mesh> const& meshes) : mMeshes(meshes) {
    for (auto& mesh : meshes) {
        mVertexCount += mesh.mVertices.size();
        mTriangleCount += mesh.mIndices.size() / 3;
    }
}

void Model::GenerateMeshBuffers(VulkanContext* context, VulkanEngine* engine) {
    for (auto& mesh : mMeshes) {
        mesh.GenerateBuffers(context, engine);
    }
}

void Model::Draw() {}

void Model::LoadModel() {
    Assimp::Importer importer {};

    const auto scene =
        importer.ReadFile(mPath, aiProcessPreset_TargetRealtime_Fast);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode) {
        // TODO: Logging
        DBG_LOG_INFO(::std::string("ERROR::ASSIMP::")
                     + importer.GetErrorString());
        return;
    }

    ProcessNode(scene->mRootNode, scene);
}

void Model::ProcessNode(aiNode* node, const aiScene* scene) {
    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        mMeshes.push_back(ProcessMesh(mesh, scene));
    }
    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(node->mChildren[i], scene);
    }
}

Mesh Model::ProcessMesh(aiMesh* mesh, const aiScene* scene) {
    ::std::vector<Vertex>   vertices;
    ::std::vector<uint32_t> indices;
    // TODO: Texture

    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        Vertex    vertex;
        glm::vec3 temp;

        // position
        if (mesh->HasPositions()) {
            temp.x = mesh->mVertices[i].x;
            if (mFlipYZ) {
                temp.y = mesh->mVertices[i].z;
                temp.z = mesh->mVertices[i].y;
            } else {
                temp.y = mesh->mVertices[i].y;
                temp.z = mesh->mVertices[i].z;
            }
            vertex.position = glm::vec4 {temp, 0.0f};
        }

        // normal
        if (mesh->HasNormals()) {
            temp.x = mesh->mNormals[i].x;
            if (mFlipYZ) {
                temp.y = mesh->mNormals[i].z;
                temp.z = mesh->mNormals[i].y;
            } else {
                temp.y = mesh->mNormals[i].y;
                temp.z = mesh->mNormals[i].z;
            }
            vertex.normal = glm::vec4 {temp, 0.0f};
        }

        // texcoords
        if (mesh->HasTextureCoords(i)) {
            glm::vec2 vec2;
            vec2.x           = mesh->mTextureCoords[0][i].x;
            vec2.y           = mesh->mTextureCoords[0][i].y;
            vertex.texcoords = vec2;
        }

        // tangents and bitangents
        if (mesh->HasTangentsAndBitangents()) {
            temp.x = mesh->mTangents[i].x;
            if (mFlipYZ) {
                temp.y = mesh->mTangents[i].z;
                temp.z = mesh->mTangents[i].y;
            } else {
                temp.y = mesh->mTangents[i].y;
                temp.z = mesh->mTangents[i].z;
            }
            vertex.tangent = glm::vec4 {temp, 0.0f};

            temp.x = mesh->mBitangents[i].x;
            if (mFlipYZ) {
                temp.y = mesh->mBitangents[i].z;
                temp.z = mesh->mBitangents[i].y;
            } else {
                temp.y = mesh->mBitangents[i].y;
                temp.z = mesh->mBitangents[i].z;
            }
            vertex.bitangent = glm::vec4 {temp, 0.0f};
        }

        vertices.push_back(vertex);
    }

    if (mesh->HasFaces()) {
        for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
            auto face = mesh->mFaces[i];
            for (uint32_t j = 0; j < face.mNumIndices; ++j) {
                indices.push_back(face.mIndices[j]);
            }
        }
    }

    // TODO: material

    mVertexCount += vertices.size();
    mTriangleCount += mesh->mNumFaces;

    return {vertices, indices};
}