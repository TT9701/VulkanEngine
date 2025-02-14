#pragma once

#include <filesystem>

#include <assimp/scene.h>
#include <vulkan/vulkan.hpp>

#include "Core/CISDIModel/CISDI_3DModelData.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Mesh.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Application;

struct MeshletPushConstants {
    IDCMCore_NS::Mat4 mModelMatrix {IDCMCore_NS::Identity4x4()};

    vk::DeviceAddress mVPBufAddr {};
    vk::DeviceAddress mVNBufAddr {};
    vk::DeviceAddress mVTBufAddr {};

    vk::DeviceAddress mMeshletBufAddr {};
    vk::DeviceAddress mMeshletTriBufAddr {};
    vk::DeviceAddress mVertOffsetBufAddr {};
    vk::DeviceAddress mMeshletOffsetBufAddr {};
    vk::DeviceAddress mMeshletTrioffsetBufAddr {};
    vk::DeviceAddress mMeshletCountBufAddr {};

    vk::DeviceAddress mBoundingBoxBufAddr {};
    vk::DeviceAddress mMeshletBoundingBoxBufAddr {};
};

struct FragmentPushConstants {
    vk::DeviceAddress mMeshMaterialIdxBufAddr {};
    vk::DeviceAddress mMaterialBufAddr {};
};

class Geometry {
public:
    Geometry(VulkanContext& context, const char* path,
             ::std::pmr::memory_resource* pMemPool, bool flipYZ = false,
             const char* output = nullptr);

    void GenerateMeshletBuffers(VulkanContext& context);

    uint32_t GetMeshCount();
    uint32_t GetVertexCount() const;
    uint32_t GetMeshletCount() const;
    uint32_t GetMeshletTriangleCount() const;

    ModelData::CISDI_3DModel const& GetCISDIModelData() const;

    ::std::span<uint32_t> GetVertexOffsets();

    GPUMeshBuffers& GetMeshBuffer();
    MeshletPushConstants GetMeshletPushContants() const;
    MeshletPushConstants* GetMeshletPushContantsPtr();
    FragmentPushConstants* GetFragmentPushConstantsPtr();

    Buffer* GetMeshTaskIndirectCmdBuffer() const;

private:
    /*
     * if "*.cisdi" model file exists, load existing "*.cisdi" model.
     * else convert 3d model file into "*.cisdi" model file. then move here from memory.
     */
    ModelData::CISDI_3DModel LoadModel(const char* output,
                                       ::std::pmr::memory_resource* pMemPool);
    void GenerateStats();

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mMeshletCount {0};
    uint32_t mMeshletTriangleCount {0};

    ::std::filesystem::path mPath;
    ::std::filesystem::path mDirectory;
    Type_STLString mName;

    ModelData::CISDI_3DModel mModelData;

    struct MeshDatas {
        Type_STLVector<uint32_t> vertexOffsets;
        Type_STLVector<uint32_t> meshletOffsets;
        Type_STLVector<uint32_t> meshletTrianglesOffsets;
        Type_STLVector<uint32_t> meshletCounts;
    };

    MeshDatas mMeshDatas;
    GPUMeshBuffers mBuffers {};

    MeshletPushConstants mMeshletConstants {};
    FragmentPushConstants mFragmentConstants {};

    Type_STLVector<vk::DrawMeshTasksIndirectCommandEXT> mMeshTaskIndirectCmds;
    SharedPtr<Buffer> mMeshTaskIndirectCmdBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core