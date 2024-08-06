#pragma once

#include <assimp/scene.h>

#include "Mesh.hpp"

class Model {
public:
    Model(::std::string const& path);

    void GenerateMeshBuffers(VulkanContext* context, VulkanEngine* engine);

    void Draw();

    ::std::vector<Mesh> const& GetMeshes() const { return mMeshes; }

private:
    void LoadModel(::std::string const& path);
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);

    // TODO: Texture

private:
    ::std::vector<Mesh> mMeshes {};
    ::std::string       mDirectory;
};