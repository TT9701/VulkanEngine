#pragma once

#include <filesystem>

#include <assimp/scene.h>

#include "Mesh.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Application;

struct IndexDrawPushConstants {
    glm::mat4 mModelMatrix {glm::mat4(1.0f)};

    vk::DeviceAddress mVertexBufferAddress {};
};

struct MeshletPushConstants {
    glm::mat4 mModelMatrix {glm::mat4(1.0f)};

    vk::DeviceAddress mVertexBufferAddress {};
    vk::DeviceAddress mMeshletBufferAddress {};
    vk::DeviceAddress mMeshletVertexBufferAddress {};
    vk::DeviceAddress mMeshletTriangleBufferAddress {};
    vk::DeviceAddress mVertexOffsetBufferAddress {};
    vk::DeviceAddress mMeshletOffsetBufferAddress {};
    vk::DeviceAddress mMeshletVertexOffsetBufferAddress {};
    vk::DeviceAddress mMeshletTriangleoffsetBufferAddress {};
    vk::DeviceAddress mMeshletCountBufferAddress {};
};

class Model {
public:
    Model(const char* path, bool flipYZ = true);

    Model(::std::span<Mesh> meshes);

    void GenerateBuffers(Context* context, Application* engine);

    void GenerateMeshletBuffers(Context* context, Application* engine);

    void Draw(vk::CommandBuffer cmd, glm::mat4 modelMatrix = glm::mat4(1.0f));

    ::std::span<Mesh> GetMeshes() { return mMeshes; }

    uint32_t GetVertexCount() const { return mVertexCount; }

    uint32_t GetIndexCount() const { return mIndexCount; }

    uint32_t GetTriangleCount() const { return mTriangleCount; }

    uint32_t GetMeshletCount() const { return mMeshletCount; }

    uint32_t GetMeshletVertexCount() const { return mMeshletVertexCount; }

    uint32_t GetMeshletTriangleCount() const { return mMeshletTriangleCount; }

    ::std::span<uint32_t> GetVertexOffsets() {
        return mMeshDatas.vertexOffsets;
    }

    ::std::span<uint32_t> GetIndexOffsets() { return mMeshDatas.indexOffsets; }

    GPUMeshBuffers& GetMeshBuffer() { return mBuffers; }

    MeshletPushConstants GetMeshletPushContants() const {
        return mMeshletConstants;
    }

    MeshletPushConstants* GetMeshletPushContantsPtr() {
        return &mMeshletConstants;
    }

    IndexDrawPushConstants GetIndexDrawPushConstants() const {
        return mIndexDrawConstants;
    }

    IndexDrawPushConstants* GetIndexDrawPushConstantsPtr() {
        return &mIndexDrawConstants;
    }

    Buffer* GetIndexedIndirectCmdBuffer() const {
        return mIndirectIndexedCmdBuffer.get();
    }

    Buffer* GetMeshTaskIndirectCmdBuffer() const {
        return mMeshTaskIndirectCmdBuffer.get();
    }

    ::std::span<vk::DrawMeshTasksIndirectCommandEXT> GetMeshTaskCmds() {
        return mMeshTaskIndirectCmds;
    }

private:
    void LoadModel();
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);

    void GenerateStats();
    void Optimize();

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mIndexCount {0};
    uint32_t mTriangleCount {0};
    uint32_t mMeshletCount {0};
    uint32_t mMeshletVertexCount {0};
    uint32_t mMeshletTriangleCount {0};

    Type_STLVector<Mesh> mMeshes {};

    ::std::filesystem::path mPath;
    ::std::filesystem::path mDirectory;
    Type_STLString mName;

    struct MeshDatas {
        Type_STLVector<uint32_t> vertexOffsets;
        Type_STLVector<uint32_t> indexOffsets;
        Type_STLVector<uint32_t> meshletOffsets;
        Type_STLVector<uint32_t> meshletVerticesOffsets;
        Type_STLVector<uint32_t> meshletTrianglesOffsets;
        Type_STLVector<uint32_t> meshletCounts;
    };

    MeshDatas mMeshDatas;
    GPUMeshBuffers mBuffers {};

    MeshletPushConstants mMeshletConstants {};
    IndexDrawPushConstants mIndexDrawConstants {};

    Type_STLVector<vk::DrawIndexedIndirectCommand> mIndirectIndexedCmds;
    Type_STLVector<vk::DrawMeshTasksIndirectCommandEXT> mMeshTaskIndirectCmds;
    SharedPtr<Buffer> mIndirectIndexedCmdBuffer;
    SharedPtr<Buffer> mMeshTaskIndirectCmdBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core