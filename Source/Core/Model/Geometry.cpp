#include "Geometry.h"

#include "Core/Application/Application.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

Geometry::Geometry(const char* path, bool flipYZ, const char* output)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(::std::filesystem::path {mPath}.remove_filename()),
      mName(mPath.stem().generic_string()) {
    LoadModel(output);
    GenerateStats();
}

void Geometry::GenerateMeshletBuffers(VulkanContext& context) {
    constexpr size_t vpSize =
        sizeof(mModelData.meshes[0].meshlets.vertices[0].positions[0]);
    constexpr size_t vnSize =
        sizeof(mModelData.meshes[0].meshlets.vertices[0].normals[0]);
    constexpr size_t vtSize =
        sizeof(mModelData.meshes[0].meshlets.vertices[0].uvs[0]);
    constexpr size_t aabbSize = sizeof(mModelData.boundingBox);

    const size_t vpBufSize = mVertexCount * vpSize;
    const size_t vnBufSize = mVertexCount * vnSize;
    const size_t vtBufSize = mVertexCount * vtSize;

    // Vertices SSBO
    {
        mBuffers.mVPBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Vertex Position").c_str(), vpBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVNBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Vertex Normal").c_str(), vnBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVTBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Vertex Texcoords").c_str(), vtBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVPBufAddr = mBuffers.mVPBuf->GetBufferDeviceAddress();
        mBuffers.mVNBufAddr = mBuffers.mVNBuf->GetBufferDeviceAddress();
        mBuffers.mVTBufAddr = mBuffers.mVTBuf->GetBufferDeviceAddress();
    }

    constexpr size_t mlSize = sizeof(mModelData.meshes[0].meshlets.infos[0]);
    constexpr size_t mlTriSize =
        sizeof(mModelData.meshes[0].meshlets.triangles[0]);
    const size_t mlBufSize = mMeshletCount * mlSize;
    const size_t mlTriBufSize = mMeshletTriangleCount * mlTriSize;

    // Meshlets buffers
    {
        mBuffers.mMeshletBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Meshlet").c_str(), mlBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletTriBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Meshlet triangles").c_str(), mlTriBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletBufAddr =
            mBuffers.mMeshletBuf->GetBufferDeviceAddress();
        mBuffers.mMeshletTriBufAddr =
            mBuffers.mMeshletTriBuf->GetBufferDeviceAddress();
    }

    uint32_t meshCount = mModelData.meshes.size();

    // bounding box
    const size_t aabbBufSize = aabbSize * (meshCount + 1 + mMeshletCount);
    {
        mBuffers.mBoundingBoxBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Bounding Box").c_str(), aabbBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mBoundingBoxBufAddr =
            mBuffers.mBoundingBoxBuf->GetBufferDeviceAddress();
    }

    // materials
    const size_t materialBufSize =
        sizeof(mModelData.materials[0].data) * mModelData.materials.size();
    {
        mBuffers.mMaterialBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Materials").c_str(), materialBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mMaterialBufAddr =
            mBuffers.mMaterialBuf->GetBufferDeviceAddress();
    }

    const size_t meshMaterialIndexBufSize = sizeof(uint32_t) * meshCount;
    {
        mBuffers.mMeshMaterialIdxBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Mesh Material Indices").c_str(),
            meshMaterialIndexBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mMeshMaterialIdxBufAddr =
            mBuffers.mMeshMaterialIdxBuf->GetBufferDeviceAddress();
    }

    // offsets data buffer *NO index*
    // vertex offsets + meshletoffsets + meshlettriangles offsets + meshlet counts
    const size_t offsetsBufSize = meshCount * 4 * sizeof(uint32_t);
    {
        mBuffers.mMeshDataBuf = context.CreateDeviceLocalBufferResource(
            (mName + " Offsets data").c_str(), offsetsBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mMeshDataBufAddr =
            mBuffers.mMeshDataBuf->GetBufferDeviceAddress();
    }

    // copy through staging buffer
    {
        auto staging = context.CreateStagingBuffer(
            "", vpBufSize + vnBufSize + vtBufSize + mlBufSize + mlTriBufSize
                    + offsetsBufSize);

        auto data = static_cast<char*>(staging->GetMapPtr());

        // vetices
        for (uint32_t i = 0; i < meshCount; ++i) {
            auto ptr = data + mMeshDatas.vertexOffsets[i] * vpSize;
            for (uint32_t j = 0; j < mModelData.meshes[i].header.meshletCount;
                 ++j) {
                auto& info = mModelData.meshes[i].meshlets.infos[j];
                auto size = info.vertexCount * vpSize;
                memcpy(
                    ptr,
                    mModelData.meshes[i].meshlets.vertices[j].positions.data(),
                    size);
                ptr += size;
            }
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            auto ptr = data + vpBufSize + mMeshDatas.vertexOffsets[i] * vnSize;
            for (uint32_t j = 0; j < mModelData.meshes[i].header.meshletCount;
                 ++j) {
                auto& info = mModelData.meshes[i].meshlets.infos[j];
                auto size = info.vertexCount * vnSize;
                memcpy(ptr,
                       mModelData.meshes[i].meshlets.vertices[j].normals.data(),
                       size);
                ptr += size;
            }
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            auto ptr = data + vpBufSize + vnBufSize
                     + mMeshDatas.vertexOffsets[i] * vtSize;
            for (uint32_t j = 0; j < mModelData.meshes[i].header.meshletCount;
                 ++j) {
                auto& info = mModelData.meshes[i].meshlets.infos[j];
                auto size = info.vertexCount * vtSize;
                memcpy(ptr,
                       mModelData.meshes[i].meshlets.vertices[j].uvs.data(),
                       size);
                ptr += size;
            }
        }

        // meshlets
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize
                       + mMeshDatas.meshletOffsets[i] * mlSize,
                   mModelData.meshes[i].meshlets.infos.data(),
                   mModelData.meshes[i].meshlets.infos.size() * mlSize);
        }

        //meshlet triangles
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize + mlBufSize
                       + mMeshDatas.meshletTrianglesOffsets[i] * mlTriSize,
                   mModelData.meshes[i].meshlets.triangles.data(),
                   mModelData.meshes[i].meshlets.triangles.size() * mlTriSize);
        }
        // offsets
        auto meshCountSize = meshCount * sizeof(meshCount);
        auto totalOffsets =
            vpBufSize + vnBufSize + vtBufSize + mlBufSize + mlTriBufSize;
        for (uint32_t i = 0; i < meshCount; ++i) {}
        {
            memcpy(data + totalOffsets, mMeshDatas.vertexOffsets.data(),
                   meshCountSize);

            memcpy(data + totalOffsets + meshCountSize,
                   mMeshDatas.meshletOffsets.data(), meshCountSize);

            memcpy(data + totalOffsets + meshCountSize * 2,
                   mMeshDatas.meshletTrianglesOffsets.data(), meshCountSize);

            memcpy(data + totalOffsets + meshCountSize * 3,
                   mMeshDatas.meshletCounts.data(), meshCountSize);
        }

        {
            auto cmd = context.CreateCmdBufToBegin(
                context.GetQueue(QueueType::Transfer));
            vk::BufferCopy vertexCopy {};
            vertexCopy.setSize(vpBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mVPBuf->GetBufferHandle(), vertexCopy);

            vertexCopy.setSize(vnBufSize).setSrcOffset(vpBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mVNBuf->GetBufferHandle(), vertexCopy);

            vertexCopy.setSize(vtBufSize).setSrcOffset(vpBufSize + vnBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mVTBuf->GetBufferHandle(), vertexCopy);

            vk::BufferCopy meshletCopy {};
            meshletCopy.setSize(mlBufSize).setSrcOffset(vpBufSize + vnBufSize
                                                        + vtBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshletBuf->GetBufferHandle(),
                            meshletCopy);

            vk::BufferCopy meshletTriCopy {};
            meshletTriCopy.setSize(mlTriBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + mlBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshletTriBuf->GetBufferHandle(),
                            meshletTriCopy);

            vk::BufferCopy offsetsCopy {};
            offsetsCopy.setSize(offsetsBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + mlBufSize
                              + mlTriBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshDataBuf->GetBufferHandle(),
                            offsetsCopy);
        }

        mMeshletConstants.mVPBufAddr = mBuffers.mVPBufAddr;
        mMeshletConstants.mVNBufAddr = mBuffers.mVNBufAddr;
        mMeshletConstants.mVTBufAddr = mBuffers.mVTBufAddr;
        mMeshletConstants.mMeshletBufAddr = mBuffers.mMeshletBufAddr;
        mMeshletConstants.mMeshletTriBufAddr = mBuffers.mMeshletTriBufAddr;

        mMeshletConstants.mVertOffsetBufAddr = mBuffers.mMeshDataBufAddr;
        mMeshletConstants.mMeshletOffsetBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize;
        mMeshletConstants.mMeshletTrioffsetBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize * 2;
        mMeshletConstants.mMeshletCountBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize * 3;
    }

    // copy aabb data
    {
        auto staging = context.CreateStagingBuffer("", aabbBufSize);
        auto data = static_cast<char*>(staging->GetMapPtr());
        memcpy(data, &mModelData.boundingBox, aabbSize);
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + aabbSize * (i + 1), &mModelData.meshes[i].boundingBox,
                   aabbSize);
        }
        char* ptr = data + aabbSize * (meshCount + 1);
        for (uint32_t i = 0; i < meshCount; ++i) {
            uint32_t size =
                mModelData.meshes[i].meshlets.boundingBoxes.size() * aabbSize;
            memcpy(ptr, mModelData.meshes[i].meshlets.boundingBoxes.data(),
                   size);
            ptr += size;
        }

        {
            auto cmd = context.CreateCmdBufToBegin(
                context.GetQueue(QueueType::Transfer));
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(aabbBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mBoundingBoxBuf->GetBufferHandle(),
                            cmdBufCopy);
        }
        mMeshletConstants.mBoundingBoxBufAddr = mBuffers.mBoundingBoxBufAddr;
        mMeshletConstants.mMeshletBoundingBoxBufAddr =
            mBuffers.mBoundingBoxBufAddr + aabbSize * (meshCount + 1);
    }

    // copy material data
    {
        auto staging = context.CreateStagingBuffer(
            "", materialBufSize + meshMaterialIndexBufSize);
        auto data = static_cast<char*>(staging->GetMapPtr());

        for (auto const& mat : mModelData.materials) {
            memcpy(data, &mat.data, sizeof(mat.data));
            data += sizeof(mat.data);
        }

        ::std::vector<uint32_t> meshMaterialIndices(meshCount);
        for (auto const& node : mModelData.nodes) {
            if (node.meshIdx == -1)
                continue;
            meshMaterialIndices[node.meshIdx] = node.materialIdx;
        }

        memcpy(data, meshMaterialIndices.data(), meshMaterialIndexBufSize);

        {
            auto cmd = context.CreateCmdBufToBegin(
                context.GetQueue(QueueType::Transfer));
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(materialBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMaterialBuf->GetBufferHandle(),
                            cmdBufCopy);
            cmdBufCopy.setSize(meshMaterialIndexBufSize)
                .setSrcOffset(materialBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshMaterialIdxBuf->GetBufferHandle(),
                            cmdBufCopy);
        }

        mFragmentConstants.mMaterialBufAddr = mBuffers.mMaterialBufAddr;
        mFragmentConstants.mMeshMaterialIdxBufAddr =
            mBuffers.mMeshMaterialIdxBufAddr;
    }

    // indirect mesh task command buffer
    {
        mMeshTaskIndirectCmdBuffer =
            context
                .CreateIndirectCmdBuffer<vk::DrawMeshTasksIndirectCommandEXT>(
                    (mName + " Mesh Task Indirect Command").c_str(),
                    mMeshTaskIndirectCmds.size());
        auto bufSize = mMeshTaskIndirectCmdBuffer->GetSize();

        auto staging = context.CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, mMeshTaskIndirectCmds.data(), bufSize);

        {
            auto cmd = context.CreateCmdBufToBegin(
                context.GetQueue(QueueType::Transfer));
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(bufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mMeshTaskIndirectCmdBuffer->GetHandle(),
                            cmdBufCopy);
        }
    }
}

uint32_t Geometry::GetMeshCount() {
    return mModelData.meshes.size();
}

uint32_t Geometry::GetVertexCount() const {
    return mVertexCount;
}

uint32_t Geometry::GetMeshletCount() const {
    return mMeshletCount;
}

uint32_t Geometry::GetMeshletTriangleCount() const {
    return mMeshletTriangleCount;
}

ModelData::CISDI_3DModel const& Geometry::GetCISDIModelData() const {
    return mModelData;
}

std::span<uint32_t> Geometry::GetVertexOffsets() {
    return mMeshDatas.vertexOffsets;
}

GPUMeshBuffers& Geometry::GetMeshBuffer() {
    return mBuffers;
}

MeshletPushConstants Geometry::GetMeshletPushContants() const {
    return mMeshletConstants;
}

MeshletPushConstants* Geometry::GetMeshletPushContantsPtr() {
    return &mMeshletConstants;
}

FragmentPushConstants* Geometry::GetFragmentPushConstantsPtr() {
    return &mFragmentConstants;
}

Buffer* Geometry::GetMeshTaskIndirectCmdBuffer() const {
    return mMeshTaskIndirectCmdBuffer.get();
}

void Geometry::LoadModel(const char* output) {
    auto modelPath = mPath.string();
    auto cisdiModelPath = modelPath + CISDI_3DModel_Subfix_Str;
    if (::std::filesystem::exists(cisdiModelPath)) {
        mModelData = IntelliDesign_NS::ModelData::Load(cisdiModelPath.c_str());
    } else {
        mModelData = IntelliDesign_NS::ModelData::Convert(modelPath.c_str(),
                                                          mFlipYZ, output);
    }
}

void Geometry::GenerateStats() {
    const auto meshCount = mModelData.meshes.size();
    mMeshDatas.vertexOffsets.reserve(meshCount);
    mMeshTaskIndirectCmds.reserve(meshCount);
    for (auto& mesh : mModelData.meshes) {
        vk::DrawMeshTasksIndirectCommandEXT meshTasksIndirectCmd {};
        meshTasksIndirectCmd
            .setGroupCountX(
                (mesh.meshlets.infos.size() + TASK_SHADER_INVOCATION_COUNT - 1)
                / TASK_SHADER_INVOCATION_COUNT)
            .setGroupCountY(1)
            .setGroupCountZ(1);
        mMeshTaskIndirectCmds.push_back(meshTasksIndirectCmd);

        mMeshDatas.vertexOffsets.push_back(mVertexCount);
        mVertexCount += mesh.header.vertexCount;

        mMeshDatas.meshletOffsets.push_back(mMeshletCount);
        mMeshletCount += mesh.meshlets.infos.size();
        mMeshDatas.meshletCounts.push_back(mesh.meshlets.infos.size());

        mMeshDatas.meshletTrianglesOffsets.push_back(mMeshletTriangleCount);
        mMeshletTriangleCount += mesh.meshlets.triangles.size();
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core