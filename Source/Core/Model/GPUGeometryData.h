#pragma once

#include <filesystem>

#include <assimp/scene.h>
#include <vulkan/vulkan.hpp>

#include "CISDIModel/CISDI_3DModelData.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Mesh.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

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

class GPUGeometryData {
public:
    GPUGeometryData(VulkanContext& context, ModelData::CISDI_3DModel const& model);

    void GenerateMeshletBuffers(VulkanContext& context,
                                ModelData::CISDI_3DModel const& model);

    Type_STLString const& GetName() const;

    uint32_t GetMeshCount() const;
    uint32_t GetVertexCount() const;
    uint32_t GetMeshletCount() const;
    uint32_t GetMeshletTriangleCount() const;

    ::std::span<uint32_t> GetVertexOffsets();

    GPUMeshBuffers& GetMeshBuffer();
    MeshletPushConstants GetMeshletPushContants() const;
    MeshletPushConstants* GetMeshletPushContantsPtr();
    FragmentPushConstants* GetFragmentPushConstantsPtr();

    Buffer* GetMeshTaskIndirectCmdBuffer() const;

private:
    void GenerateStats(ModelData::CISDI_3DModel const& model);

    // TODO: Texture

private:
    bool mFlipYZ;

    uint32_t mVertexCount {0};
    uint32_t mMeshCount {0};
    uint32_t mMeshletCount {0};
    uint32_t mMeshletTriangleCount {0};

    Type_STLString mName;

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