#pragma once

#include <assimp/scene.h>

#include "Mesh.hpp"

class VulkanContext;
class VulkanEngine;

struct PushConstants {
    glm::mat4 mModelMatrix {glm::mat4(1.0f)};
    vk::DeviceAddress mVertexBufferAddress {};
};

class Model {
public:
    Model(const char* path, bool flipYZ = true);

    Model(::std::span<Mesh> meshes);

    void GenerateBuffers(VulkanContext* context, VulkanEngine* engine);

    void Draw(vk::CommandBuffer cmd, glm::mat4 modelMatrix = glm::mat4(1.0f));

    ::std::span<Mesh> GetMeshes() { return mMeshes; }

    uint32_t GetVertexCount() const { return mVertexCount; }

    uint32_t GetIndexCount() const { return mIndexCount; }

    uint32_t GetTriangleCount() const { return mTriangleCount; }

    ::std::span<uint32_t> GetVertexOffsets() { return mOffsets.vertexOffsets; }

    ::std::span<uint32_t> GetIndexOffsets() { return mOffsets.indexOffsets; }

    GPUMeshBuffers GetMeshBuffer() const { return mBuffers; }

    PushConstants GetPushContants() const { return mConstants; }

    VulkanRenderResource* GetIndirectCmdBuffer() const {
        return mIndirectCmdBuffer.get();
    }

private:
    void LoadModel();
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mIndexCount {0};
    uint32_t mTriangleCount {0};

    ::std::vector<Mesh> mMeshes {};

    ::std::filesystem::path mPath;
    ::std::filesystem::path mDirectory;
    ::std::string mName;

    struct Offsets {
        ::std::vector<uint32_t> vertexOffsets;
        ::std::vector<uint32_t> indexOffsets;
    };

    Offsets mOffsets;
    GPUMeshBuffers mBuffers {};

    PushConstants mConstants {};

    ::std::vector<vk::DrawIndexedIndirectCommand> mIndirectCmds;
    SharedPtr<VulkanRenderResource> mIndirectCmdBuffer;
};