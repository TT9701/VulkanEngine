#pragma once

#include <assimp/scene.h>

#include "Mesh.hpp"

class Model {
public:
    Model(::std::string const& path, bool flipYZ = true);

    Model(::std::vector<Mesh> const& meshes);

    void GenerateMeshBuffers(VulkanContext* context, VulkanEngine* engine);

    void Draw();

    ::std::vector<Mesh> const& GetMeshes() const { return mMeshes; }

private:
    void LoadModel();
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mTriangleCount {0};

    ::std::vector<Mesh> mMeshes {};

    ::std::string mPath;
    ::std::string mDirectory;
    ::std::string mName;
};