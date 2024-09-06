#include "Model.hpp"

#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
// #include <assimp/DefaultLogger.hpp>

#include "Core/Utilities/VulkanUtilities.hpp"
#include "Core/Vulkan/EngineCore.hpp"
#include "Core/Vulkan/Manager/Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

Model::Model(const char* path, bool flipYZ)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(::std::filesystem::path {mPath}.remove_filename()),
      mName(mPath.stem().generic_string()) {
    LoadModel();
    Optimize();
    GenerateStats();
}

Model::Model(::std::span<Mesh> meshes) : mMeshes(meshes.begin(), meshes.end()) {
    GenerateStats();
}

void Model::GenerateBuffers(Context* context, EngineCore* engine) {
    // Vertex & index buffer
    {
        const size_t vertexSize = sizeof(mMeshes[0].mVertices[0]);
        const size_t indexSize = sizeof(mMeshes[0].mIndices[0]);

        const size_t vertexBufferSize = mVertexCount * vertexSize;
        const size_t indexBufferSize = mIndexCount * indexSize;

        mBuffers.mVertexBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex").c_str(), vertexBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mIndexBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Index").c_str(), indexBufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer
                | vk::BufferUsageFlagBits::eTransferDst);

        mBuffers.mVertexBufferAddress =
            mBuffers.mVertexBuffer->GetBufferDeviceAddress();

        auto staging = context->CreateStagingBuffer(
            "", vertexBufferSize + indexBufferSize);

        void* data = staging->GetMapPtr();
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy((Vertex*)data + mMeshDatas.vertexOffsets[i],
                   mMeshes[i].mVertices.data(),
                   mMeshes[i].mVertices.size() * vertexSize);
        }
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy((uint32_t*)((char*)data + vertexBufferSize)
                       + mMeshDatas.indexOffsets[i],
                   mMeshes[i].mIndices.data(),
                   mMeshes[i].mIndices.size() * indexSize);
        }

        engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy vertexCopy {};
            vertexCopy.setSize(vertexBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mVertexBuffer->GetBufferHandle(),
                           vertexCopy);

            vk::BufferCopy indexCopy {};
            indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mIndexBuffer->GetBufferHandle(), indexCopy);
        });

        mConstants.mVertexBufferAddress = mBuffers.mVertexBufferAddress;
    }

    // indirect indexed command buffer
    {
        auto bufSize = sizeof(vk::DrawIndexedIndirectCommand)
                     * mIndirectIndexedCmds.size();
        mIndirectIndexedCmdBuffer = context->CreateIndirectCmdBuffer(
            (mName + " Indirect Command").c_str(), bufSize);

        auto staging = context->CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, mIndirectIndexedCmds.data(), bufSize);

        engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(bufSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mIndirectIndexedCmdBuffer->GetHandle(),
                           cmdBufCopy);
        });
    }
}

void Model::GenerateMeshletBuffers(Context* context, EngineCore* engine) {
    const size_t vertexSize = sizeof(mMeshes[0].mVertices[0]);
    const size_t vertexBufferSize = mVertexCount * vertexSize;

    // Vertices SSBO
    {
        mBuffers.mVertexBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex").c_str(), vertexBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVertexBufferAddress =
            mBuffers.mVertexBuffer->GetBufferDeviceAddress();
    }
    const size_t meshletSize = sizeof(mMeshes[0].mMeshlets[0]);
    const size_t meshletVerticesSize = sizeof(mMeshes[0].mMeshletVertices[0]);
    const size_t meshletTrianglesSize = sizeof(mMeshes[0].mMeshletTriangles[0]);
    const size_t meshletBufferSize = mMeshletCount * meshletSize;
    const size_t meshletVerticesBufferSize =
        mMeshletVertexCount * meshletVerticesSize;
    const size_t meshletTrianglesBufferSize =
        mMeshletTriangleCount * meshletTrianglesSize;

    // Meshlets buffers
    {
        mBuffers.mMeshletBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet").c_str(), meshletBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletVertBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet vertices").c_str(), meshletVerticesBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletTriBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet triangles").c_str(), meshletTrianglesBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletBufferAddress =
            mBuffers.mMeshletBuffer->GetBufferDeviceAddress();
        mBuffers.mMeshletVertBufferAddress =
            mBuffers.mMeshletVertBuffer->GetBufferDeviceAddress();
        mBuffers.mMeshletTriBufferAddress =
            mBuffers.mMeshletTriBuffer->GetBufferDeviceAddress();
    }

    // offsets data buffer *NO index*
    // vertex offsets + meshletoffsets + meshletVertices offsets + meshlettriangles offsets + meshlet counts
    const size_t offsetsBufferSize = mMeshes.size() * 5 * sizeof(uint32_t);
    {
        mBuffers.mMeshDataBuffer = context->CreateDeviceLocalBufferResource(
            (mName + " Offsets data").c_str(), offsetsBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mMeshDataBufferAddress =
            mBuffers.mMeshDataBuffer->GetBufferDeviceAddress();
    }

    // copy through staging buffer
    {
        auto staging = context->CreateStagingBuffer(
            "", vertexBufferSize + meshletBufferSize + meshletVerticesBufferSize
                    + meshletTrianglesBufferSize + offsetsBufferSize);

        void* data = staging->GetMapPtr();
        // vetices
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy((Vertex*)data + mMeshDatas.vertexOffsets[i],
                   mMeshes[i].mVertices.data(),
                   mMeshes[i].mVertices.size() * vertexSize);
        }
        // meshlets
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy((meshopt_Meshlet*)((char*)data + vertexBufferSize)
                       + mMeshDatas.meshletOffsets[i],
                   mMeshes[i].mMeshlets.data(),
                   mMeshes[i].mMeshlets.size() * meshletSize);
        }
        // meshlet vertices
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy(
                (uint32_t*)((char*)data + vertexBufferSize + meshletBufferSize)
                    + mMeshDatas.meshletVerticesOffsets[i],
                mMeshes[i].mMeshletVertices.data(),
                mMeshes[i].mMeshletVertices.size() * meshletVerticesSize);
        }
        //meshlet triangles
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {
            memcpy((uint8_t*)((char*)data + vertexBufferSize + meshletBufferSize
                              + meshletVerticesBufferSize)
                       + mMeshDatas.meshletTrianglesOffsets[i],
                   mMeshes[i].mMeshletTriangles.data(),
                   mMeshes[i].mMeshletTriangles.size() * meshletTrianglesSize);
        }
        // offsets
        for (uint32_t i = 0; i < mMeshes.size(); ++i) {}
        {
            memcpy((char*)data + vertexBufferSize + meshletBufferSize
                       + meshletVerticesBufferSize + meshletTrianglesBufferSize,
                   mMeshDatas.vertexOffsets.data(),
                   sizeof(uint32_t) * mMeshes.size());

            memcpy((char*)data + vertexBufferSize + meshletBufferSize
                       + meshletVerticesBufferSize + meshletTrianglesBufferSize
                       + sizeof(uint32_t) * mMeshes.size(),
                   mMeshDatas.meshletOffsets.data(),
                   sizeof(uint32_t) * mMeshes.size());

            memcpy((char*)data + vertexBufferSize + meshletBufferSize
                       + meshletVerticesBufferSize + meshletTrianglesBufferSize
                       + sizeof(uint32_t) * mMeshes.size() * 2,
                   mMeshDatas.meshletVerticesOffsets.data(),
                   sizeof(uint32_t) * mMeshes.size());

            memcpy((char*)data + vertexBufferSize + meshletBufferSize
                       + meshletVerticesBufferSize + meshletTrianglesBufferSize
                       + sizeof(uint32_t) * mMeshes.size() * 3,
                   mMeshDatas.meshletTrianglesOffsets.data(),
                   sizeof(uint32_t) * mMeshes.size());

            memcpy((char*)data + vertexBufferSize + meshletBufferSize
                       + meshletVerticesBufferSize + meshletTrianglesBufferSize
                       + sizeof(uint32_t) * mMeshes.size() * 4,
                   mMeshDatas.meshletCounts.data(),
                   sizeof(uint32_t) * mMeshes.size());
        }

        engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy vertexCopy {};
            vertexCopy.setSize(vertexBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mVertexBuffer->GetBufferHandle(),
                           vertexCopy);

            vk::BufferCopy meshletCopy {};
            meshletCopy.setSize(meshletBufferSize)
                .setSrcOffset(vertexBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mMeshletBuffer->GetBufferHandle(),
                           meshletCopy);

            vk::BufferCopy meshletVertCopy {};
            meshletVertCopy.setSize(meshletVerticesBufferSize)
                .setSrcOffset(vertexBufferSize + meshletBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mMeshletVertBuffer->GetBufferHandle(),
                           meshletVertCopy);

            vk::BufferCopy meshletTriCopy {};
            meshletTriCopy.setSize(meshletTrianglesBufferSize)
                .setSrcOffset(vertexBufferSize + meshletBufferSize
                              + meshletVerticesBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mMeshletTriBuffer->GetBufferHandle(),
                           meshletTriCopy);

            vk::BufferCopy offsetsCopy {};
            offsetsCopy.setSize(offsetsBufferSize)
                .setSrcOffset(vertexBufferSize + meshletBufferSize
                              + meshletVerticesBufferSize
                              + meshletTrianglesBufferSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mBuffers.mMeshDataBuffer->GetBufferHandle(),
                           offsetsCopy);
        });

        mConstants.mVertexBufferAddress = mBuffers.mVertexBufferAddress;
        mConstants.mMeshletBufferAddress = mBuffers.mMeshletBufferAddress;
        mConstants.mMeshletVertexBufferAddress =
            mBuffers.mMeshletVertBufferAddress;
        mConstants.mMeshletTriangleBufferAddress =
            mBuffers.mMeshletTriBufferAddress;

        mConstants.mVertexOffsetBufferAddress = mBuffers.mMeshDataBufferAddress;
        mConstants.mMeshletOffsetBufferAddress =
            mBuffers.mMeshDataBufferAddress + sizeof(uint32_t) * mMeshes.size();
        mConstants.mMeshletVertexOffsetBufferAddress =
            mBuffers.mMeshDataBufferAddress
            + sizeof(uint32_t) * mMeshes.size() * 2;
        mConstants.mMeshletTriangleoffsetBufferAddress =
            mBuffers.mMeshDataBufferAddress
            + sizeof(uint32_t) * mMeshes.size() * 3;
        mConstants.mMeshletCountBufferAddress =
            mBuffers.mMeshDataBufferAddress
            + sizeof(uint32_t) * mMeshes.size() * 4;
    }

    // indirect mesh task command buffer
    {
        auto bufSize = sizeof(vk::DrawMeshTasksIndirectCommandEXT)
                     * mMeshTaskIndirectCmds.size();
        mMeshTaskIndirectCmdBuffer = context->CreateIndirectCmdBuffer(
            (mName + " Mesh Task Indirect Command").c_str(), bufSize);

        auto staging = context->CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, mMeshTaskIndirectCmds.data(), bufSize);

        engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(bufSize);
            cmd.copyBuffer(staging->GetHandle(),
                           mMeshTaskIndirectCmdBuffer->GetHandle(),
                           cmdBufCopy);
        });
    }
}

void Model::Draw(vk::CommandBuffer cmd, glm::mat4 modelMatrix) {
    // cmd.bindIndexBuffer(mBuffers.mIndexBuffer->GetHandle(), 0,
    //                     vk::IndexType::eUint32);
    //
    // mConstants.mModelMatrix = modelMatrix;
    //
    // cmd.pushConstants(mPipelineManager->GetLayoutHandle("Triangle_Layout"),
    //                   vk::ShaderStageFlagBits::eVertex, 0, sizeof(pushContants),
    //                   &pushContants);
}

void Model::LoadModel() {
    Assimp::Importer importer {};

    const auto scene = importer.ReadFile(mPath.generic_string().c_str(),
                                         aiProcessPreset_TargetRealtime_Fast);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode) {
        // TODO: Logging
        DBG_LOG_INFO(
            (Type_STLString("ERROR::ASSIMP::") + importer.GetErrorString())
                .c_str());
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
    Type_STLVector<Vertex> vertices;
    Type_STLVector<uint32_t> indices;
    // TODO: Texture

    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        Vertex vertex;
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
            vertex.position = glm::vec4 {temp, 1.0f};
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
            vec2.x = mesh->mTextureCoords[0][i].x;
            vec2.y = mesh->mTextureCoords[0][i].y;
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

    return {vertices, indices};
}

void Model::GenerateStats() {
    mMeshDatas.vertexOffsets.reserve(mMeshes.size());
    mMeshDatas.indexOffsets.reserve(mMeshes.size());
    mIndirectIndexedCmds.reserve(mMeshes.size());
    mMeshTaskIndirectCmds.reserve(mMeshes.size());
    for (auto& mesh : mMeshes) {
        vk::DrawIndexedIndirectCommand indexedCmd {};
        indexedCmd.setFirstInstance(0)
            .setInstanceCount(1)
            .setFirstIndex(mIndexCount)
            .setIndexCount(mesh.mIndices.size())
            .setVertexOffset(mVertexCount);
        mIndirectIndexedCmds.push_back(indexedCmd);

        vk::DrawMeshTasksIndirectCommandEXT meshTasksIndirectCmd {};
        meshTasksIndirectCmd
            .setGroupCountX(
                (mesh.mMeshlets.size() + TASK_SHADER_INVOCATION_COUNT - 1)
                / TASK_SHADER_INVOCATION_COUNT)
            .setGroupCountY(1)
            .setGroupCountZ(1);
        mMeshTaskIndirectCmds.push_back(meshTasksIndirectCmd);

        mMeshDatas.vertexOffsets.push_back(mVertexCount);
        mVertexCount += mesh.mVertices.size();

        mMeshDatas.indexOffsets.push_back(mIndexCount);
        mIndexCount += mesh.mIndices.size();

        mMeshDatas.meshletOffsets.push_back(mMeshletCount);
        mMeshletCount += mesh.mMeshlets.size();
        mMeshDatas.meshletCounts.push_back(mesh.mMeshlets.size());

        mMeshDatas.meshletVerticesOffsets.push_back(mMeshletVertexCount);
        mMeshletVertexCount += mesh.mMeshletVertices.size();

        mMeshDatas.meshletTrianglesOffsets.push_back(mMeshletTriangleCount);
        mMeshletTriangleCount += mesh.mMeshletTriangles.size();
    }
    mTriangleCount = mIndexCount / 3;
}

void Model::Optimize() {
    for (auto& mesh : mMeshes) {
        size_t indexCount = mesh.mIndices.size();
        size_t vertexCount = mesh.mVertices.size();

        // Type_STLVector<Vertex> optimizedVertices;
        // Type_STLVector<uint32_t> optimizedIndices;
        //
        // Type_STLVector<uint32_t> remap(indexCount);
        // vertexCount = meshopt_generateVertexRemap(
        //     remap.data(), mesh.mIndices.data(), indexCount,
        //     mesh.mVertices.data(), mesh.mVertices.size(),
        //     sizeof(mesh.mVertices[0]));
        //
        // optimizedIndices.resize(indexCount);
        // optimizedVertices.resize(vertexCount);
        // meshopt_remapIndexBuffer(optimizedIndices.data(), mesh.mIndices.data(),
        //                          indexCount, remap.data());
        // meshopt_remapVertexBuffer(optimizedVertices.data(),
        //                           mesh.mVertices.data(), vertexCount,
        //                           sizeof(mesh.mVertices[0]), remap.data());
        //
        // mesh.mVertices = optimizedVertices;
        // mesh.mIndices = optimizedIndices;

        meshopt_optimizeVertexCache(mesh.mIndices.data(), mesh.mIndices.data(),
                                    indexCount, vertexCount);

        meshopt_optimizeOverdraw(
            mesh.mIndices.data(), mesh.mIndices.data(), indexCount,
            (const float*)(&mesh.mVertices[0] + offsetof(Vertex, position)),
            vertexCount, sizeof(Vertex), 1.05f);

        // meshopt_optimizeVertexFetch(mesh.mVertices.data(), mesh.mIndices.data(),
        //                             indexCount, mesh.mVertices.data(),
        //                             vertexCount, sizeof(Vertex));

        const size_t maxVertices = 64;
        const size_t maxTriangles = 124;
        const float coneWeight = 0.0f;

        size_t maxMeshlets = meshopt_buildMeshletsBound(
            mesh.mIndices.size(), maxVertices, maxTriangles);

        mesh.mMeshlets.resize(maxMeshlets);
        mesh.mMeshletVertices.resize(maxMeshlets * maxVertices);
        mesh.mMeshletTriangles.resize(maxMeshlets * maxTriangles * 3);

        size_t meshletCount = meshopt_buildMeshlets(
            mesh.mMeshlets.data(), mesh.mMeshletVertices.data(),
            mesh.mMeshletTriangles.data(), mesh.mIndices.data(),
            mesh.mIndices.size(),
            (const float*)(&mesh.mVertices[0] + offsetof(Vertex, position)),
            mesh.mVertices.size(), sizeof(Vertex), maxVertices, maxTriangles,
            coneWeight);

        const meshopt_Meshlet& last = mesh.mMeshlets[meshletCount - 1];

        mesh.mMeshletVertices.resize(last.vertex_offset + last.vertex_count);
        mesh.mMeshletTriangles.resize(last.triangle_offset
                                      + ((last.triangle_count * 3 + 3) & ~3));
        mesh.mMeshlets.resize(meshletCount);

        for (auto& meshlet : mesh.mMeshlets) {
            meshopt_optimizeMeshlet(
                &mesh.mMeshletVertices[meshlet.vertex_offset],
                &mesh.mMeshletTriangles[meshlet.triangle_offset],
                meshlet.triangle_count, meshlet.vertex_count);
        }
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core