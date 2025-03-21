#pragma once

#include <filesystem>

#include <assimp/scene.h>
#include <vulkan/vulkan.hpp>

#include "CISDIModel/CISDI_3DModelData.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "Mesh.h"

namespace IntelliDesign_NS::Vulkan::Core {

namespace IDCMCore_NS = CMCore_NS;

class VulkanContext;

struct MeshletPushConstants {
    IDCMCore_NS::Mat4 mModelMatrix {
        IDCMCore_NS::Mat4 {IDCMCore_NS::MatrixScaling(0.01f, 0.01f, 0.01f)}};

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

    vk::DeviceAddress mMeshMaterialIdxBufAddr {};
    vk::DeviceAddress mMaterialBufAddr {};
};

class GPUGeometryData {
public:
    GPUGeometryData(VulkanContext& context,
                    ModelData::CISDI_3DModel const& model,
                    uint32_t maxMeshCountPerDGCSequence);

    void GenerateMeshletBuffers(VulkanContext& context,
                                ModelData::CISDI_3DModel const& model,
                                uint32_t maxMeshCount);

    Type_STLString const& GetName() const;

    MeshletPushConstants GetMeshletPushContants(uint32_t idx = 0) const;

    vk::DrawIndirectCountIndirectCommandEXT GetDrawIndirectCmdBufInfo(
        uint32_t idx = 0) const;

    uint32_t GetSequenceCount() const;

private:
    void GenerateStats(ModelData::CISDI_3DModel const& model,
                       uint32_t maxMeshCount);

    // TODO: Texture

private:
    Type_STLString mName;
    uint32_t mSequenceCount;

    struct MeshDatas {
        uint32_t mVertexCount {0};
        uint32_t mMeshCount {0};
        uint32_t mMeshletCount {0};
        uint32_t mMeshletTriangleCount {0};

        Type_STLVector<uint32_t> vertexOffsets;
        Type_STLVector<uint32_t> meshletOffsets;
        Type_STLVector<uint32_t> meshletTrianglesOffsets;
        Type_STLVector<uint32_t> meshletCounts;
    };

    Type_STLVector<MeshDatas> mMeshDatas;
    Type_STLVector<Type_STLVector<vk::DrawMeshTasksIndirectCommandEXT>>
        mMeshTaskIndirectCmds;

    Type_STLVector<GPUMeshBuffers> mBuffers {};
    Type_STLVector<MeshletPushConstants> mMeshletConstants {};

    Type_STLVector<SharedPtr<Buffer>> mMeshTaskIndirectCmdBuffer;
};

}  // namespace IntelliDesign_NS::Vulkan::Core