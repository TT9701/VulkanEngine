#include "Model.hpp"

#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
// #include <assimp/DefaultLogger.hpp>

#include "Core/Utilities/Logger.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/VulkanCore/VulkanContext.hpp"
#include "Core/VulkanCore/VulkanEngine.hpp"

Model::Model(std::string const& path, bool flipYZ)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(Utils::GetDirectory(path)),
      mName(Utils::GetFileName(path)) {
    LoadModel();
}

Model::Model(std::vector<Mesh> const& meshes) : mMeshes(meshes) {
    mOffsets.vertexOffsets.reserve(mMeshes.size());
    mOffsets.indexOffsets.reserve(mMeshes.size());
    for (auto& mesh : mMeshes) {
        mOffsets.vertexOffsets.push_back(mVertexCount);
        mVertexCount += mesh.mVertices.size();
        mOffsets.indexOffsets.push_back(mIndexCount);
        mIndexCount += mesh.mIndices.size();
    }
    mTriangleCount = mIndexCount / 3;
}

void Model::GenerateMeshBuffers(VulkanContext* context, VulkanEngine* engine) {
    const size_t vertexSize = sizeof(mMeshes[0].mVertices[0]);
    const size_t indexSize  = sizeof(mMeshes[0].mIndices[0]);

    const size_t vertexBufferSize = mVertexCount * vertexSize;
    const size_t indexBufferSize  = mIndexCount * indexSize;

    mBuffers.mVertexBuffer = context->CreatePersistentBuffer(
        vertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer
                              | vk::BufferUsageFlagBits::eTransferDst
                              | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mBuffers.mIndexBuffer = context->CreatePersistentBuffer(
        indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer
                             | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(mBuffers.mVertexBuffer->GetHandle());

    mBuffers.mVertexBufferAddress =
        context->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    auto staging =
        context->CreateStagingBuffer(vertexBufferSize + indexBufferSize);

    void* data = staging->GetAllocationInfo().pMappedData;
    for (uint32_t i = 0; i < mMeshes.size(); ++i) {
        memcpy((Vertex*)data + mOffsets.vertexOffsets[i],
               mMeshes[i].mVertices.data(),
               mMeshes[i].mVertices.size() * vertexSize);
    }
    for (uint32_t i = 0; i < mMeshes.size(); ++i) {
        memcpy((uint32_t*)((char*)data + vertexBufferSize)
                   + mOffsets.indexOffsets[i],
               mMeshes[i].mIndices.data(),
               mMeshes[i].mIndices.size() * indexSize);
    }

    engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(),
                       mBuffers.mVertexBuffer->GetHandle(), vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), mBuffers.mIndexBuffer->GetHandle(),
                       indexCopy);
    });

    mConstants.mVertexBufferAddress = mBuffers.mVertexBufferAddress;
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