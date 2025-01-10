#pragma once

#include <filesystem>

#include <assimp/scene.h>
#include <vulkan/vulkan.hpp>

#include "Core/CISDIModel/CISDI_3DModelData.h"
#include "Mesh.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Application;

struct MeshletPushConstants {
    glm::mat4 mModelMatrix {glm::mat4(1.0f)};

    vk::DeviceAddress mVPBufAddr {};
    vk::DeviceAddress mVNBufAddr {};
    vk::DeviceAddress mVTBufAddr {};

    vk::DeviceAddress mMeshletBufAddr {};
    vk::DeviceAddress mMeshletVertBufAddr {};
    vk::DeviceAddress mMeshletTriBufAddr {};
    vk::DeviceAddress mVertOffsetBufAddr {};
    vk::DeviceAddress mMeshletOffsetBufAddr {};
    vk::DeviceAddress mMeshletVertOffsetBufAddr {};
    vk::DeviceAddress mMeshletTrioffsetBufAddr {};
    vk::DeviceAddress mMeshletCountBufAddr {};

    vk::DeviceAddress mBoundingBoxBufAddr {};
};

class Geometry {
public:
    Geometry(const char* path, bool flipYZ = true,
             const char* output = nullptr);

    void GenerateMeshletBuffers(VulkanContext& context);

    uint32_t GetMeshCount();
    uint32_t GetVertexCount() const;
    uint32_t GetMeshletCount() const;
    uint32_t GetMeshletVertexCount() const;
    uint32_t GetMeshletTriangleCount() const;

    ModelData::CISDI_3DModel const& GetCISDIModelData() const;

    ::std::span<uint32_t> GetVertexOffsets();

    GPUMeshBuffers& GetMeshBuffer();
    MeshletPushConstants GetMeshletPushContants() const;
    MeshletPushConstants* GetMeshletPushContantsPtr();

    Buffer* GetMeshTaskIndirectCmdBuffer() const;

private:
    /*
     * if "*.cisdi" model file exists, load existing "*.cisdi" model.
     * else convert 3d model file into "*.cisdi" model file. then move here from memory.
     */
    void LoadModel(const char* output);
    void GenerateStats();

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mMeshletCount {0};
    uint32_t mMeshletVertexCount {0};
    uint32_t mMeshletTriangleCount {0};

    ModelData::CISDI_3DModel mModelData;

    ::std::filesystem::path mPath;
    ::std::filesystem::path mDirectory;
    Type_STLString mName;

    struct MeshDatas {
        Type_STLVector<uint32_t> vertexOffsets;
        Type_STLVector<uint32_t> meshletOffsets;
        Type_STLVector<uint32_t> meshletVerticesOffsets;
        Type_STLVector<uint32_t> meshletTrianglesOffsets;
        Type_STLVector<uint32_t> meshletCounts;
    };

    MeshDatas mMeshDatas;
    GPUMeshBuffers mBuffers {};

    MeshletPushConstants mMeshletConstants {};

    Type_STLVector<vk::DrawMeshTasksIndirectCommandEXT> mMeshTaskIndirectCmds;
    SharedPtr<Buffer> mMeshTaskIndirectCmdBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core