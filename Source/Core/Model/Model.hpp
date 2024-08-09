#pragma once

#include <assimp/scene.h>

#include "Mesh.hpp"

class VulkanContext;
class VulkanEngine;

struct PushConstants {
    glm::mat4         mModelMatrix {glm::mat4(1.0f)};
    vk::DeviceAddress mVertexBufferAddress {};
};

class Model {
public:
    Model(::std::string const& path, bool flipYZ = true);

    Model(::std::vector<Mesh> const& meshes);

    void GenerateMeshBuffers(VulkanContext* context, VulkanEngine* engine);

    void Draw();

    ::std::vector<Mesh> const& GetMeshes() const { return mMeshes; }

    uint32_t GetVertexCount() const { return mVertexCount; }

    uint32_t GetIndexCount() const { return mIndexCount; }

    uint32_t GetTriangleCount() const { return mTriangleCount; }

    ::std::vector<uint32_t> const& GetVertexOffsets() const {
        return mOffsets.vertexOffsets;
    }

    ::std::vector<uint32_t> const& GetIndexOffsets() const {
        return mOffsets.indexOffsets;
    }

    GPUMeshBuffers GetBuffers() const { return mBuffers; }

    PushConstants GetPushContants() const { return mConstants; }

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

    ::std::string mPath;
    ::std::string mDirectory;
    ::std::string mName;

    struct Offsets {
        ::std::vector<uint32_t> vertexOffsets;
        ::std::vector<uint32_t> indexOffsets;
    };

    Offsets        mOffsets;
    GPUMeshBuffers mBuffers {};

    PushConstants mConstants {};
};