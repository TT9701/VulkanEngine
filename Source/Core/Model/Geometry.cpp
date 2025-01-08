#include "Geometry.h"

#include "Core/Application/Application.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

Geometry::Geometry(const char* path, bool flipYZ, const char* output,
                   bool optimizeMesh, bool buildMeshlet, bool optimizeMeshlet)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(::std::filesystem::path {mPath}.remove_filename()),
      mName(mPath.stem().generic_string()) {
    LoadModel(output, optimizeMesh, buildMeshlet, optimizeMeshlet);
    GenerateStats();
}

void Geometry::GenerateBuffers(VulkanContext* context) {
    // Vertex & index buffer
    {
        constexpr size_t vpSize =
            sizeof(mModelData.meshes[0].vertices.positions[0]);
        constexpr size_t vnSize =
            sizeof(mModelData.meshes[0].vertices.normals[0]);
        constexpr size_t vtSize = sizeof(mModelData.meshes[0].vertices.uvs[0]);

        constexpr size_t idxSize = sizeof(mModelData.meshes[0].indices[0]);

        const size_t vpBufSize = mVertexCount * vpSize;
        const size_t vnBufSize = mVertexCount * vnSize;
        const size_t vtBufSize = mVertexCount * vtSize;

        const size_t idxBufSize = mIndexCount * idxSize;

        mBuffers.mVPBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Position").c_str(), vpBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVNBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Normal").c_str(), vnBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVTBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Texcoords").c_str(), vtBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mIdxBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Index").c_str(), idxBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        uint32_t meshCount = mModelData.meshes.size();

        uint32_t offsetBufSize = meshCount * sizeof(uint32_t);
        mBuffers.mMeshDataBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Offsets data").c_str(), offsetBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVPBufAddr = mBuffers.mVPBuf->GetBufferDeviceAddress();
        mBuffers.mVNBufAddr = mBuffers.mVNBuf->GetBufferDeviceAddress();
        mBuffers.mVTBufAddr = mBuffers.mVTBuf->GetBufferDeviceAddress();
        mBuffers.mIdxBufAddr = mBuffers.mIdxBuf->GetBufferDeviceAddress();
        mBuffers.mMeshDataBufAddr =
            mBuffers.mMeshDataBuf->GetBufferDeviceAddress();

        auto staging = context->CreateStagingBuffer(
            "", vpBufSize + vnBufSize + vtBufSize + idxBufSize + offsetBufSize);

        auto data = static_cast<char*>(staging->GetMapPtr());
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + mMeshDatas.vertexOffsets[i] * vpSize,
                   mModelData.meshes[i].vertices.positions.data(),
                   mModelData.meshes[i].vertices.positions.size() * vpSize);
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + mMeshDatas.vertexOffsets[i] * vnSize,
                   mModelData.meshes[i].vertices.normals.data(),
                   mModelData.meshes[i].vertices.normals.size() * vnSize);
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize
                       + mMeshDatas.vertexOffsets[i] * vtSize,
                   mModelData.meshes[i].vertices.uvs.data(),
                   mModelData.meshes[i].vertices.uvs.size() * vtSize);
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize
                       + mMeshDatas.indexOffsets[i] * idxSize,
                   mModelData.meshes[i].indices.data(),
                   mModelData.meshes[i].indices.size() * idxSize);
        }

        memcpy(data + vpBufSize + vnBufSize + vtBufSize + idxBufSize,
               mMeshDatas.vertexOffsets.data(), offsetBufSize);

        {
            auto cmd = context->CreateCmdBufToBegin(
                context->GetQueue(QueueType::Transfer));
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

            vk::BufferCopy indexCopy {};
            indexCopy.setSize(idxBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mIdxBuf->GetBufferHandle(), indexCopy);

            vk::BufferCopy offsetCopy {};
            offsetCopy.setSize(offsetBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + idxBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshDataBuf->GetBufferHandle(),
                            offsetCopy);
        }

        mIndexDrawConstants.mVPBufAddr = mBuffers.mVPBufAddr;
        mIndexDrawConstants.mVNBufAddr = mBuffers.mVNBufAddr;
        mIndexDrawConstants.mVTBufAddr = mBuffers.mVTBufAddr;
        mIndexDrawConstants.mIdxBufAddr = mBuffers.mIdxBufAddr;
        mIndexDrawConstants.mVertOffsetBufAddr = mBuffers.mMeshDataBufAddr;
    }

    // indirect command buffer
    {
        mIndirectCmdBuffer =
            context->CreateIndirectCmdBuffer<vk::DrawIndirectCommand>(
                (mName + " Indirect Command").c_str(), mIndirectCmds.size());
        auto bufSize = mIndirectCmdBuffer->GetSize();

        auto staging = context->CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, mIndirectCmds.data(), bufSize);

        {
            auto cmd = context->CreateCmdBufToBegin(
                context->GetQueue(QueueType::Transfer));
            vk::BufferCopy cmdBufCopy {};
            cmdBufCopy.setSize(bufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mIndirectCmdBuffer->GetHandle(), cmdBufCopy);
        }
    }
}

void Geometry::GenerateMeshletBuffers(VulkanContext* context) {
    constexpr size_t vpSize =
        sizeof(mModelData.meshes[0].vertices.positions[0]);
    constexpr size_t vnSize = sizeof(mModelData.meshes[0].vertices.normals[0]);
    constexpr size_t vtSize = sizeof(mModelData.meshes[0].vertices.uvs[0]);

    const size_t vpBufSize = mVertexCount * vpSize;
    const size_t vnBufSize = mVertexCount * vnSize;
    const size_t vtBufSize = mVertexCount * vtSize;

    // Vertices SSBO
    {
        mBuffers.mVPBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Position").c_str(), vpBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVNBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Normal").c_str(), vnBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVTBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Vertex Texcoords").c_str(), vtBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mVPBufAddr = mBuffers.mVPBuf->GetBufferDeviceAddress();
        mBuffers.mVNBufAddr = mBuffers.mVNBuf->GetBufferDeviceAddress();
        mBuffers.mVTBufAddr = mBuffers.mVTBuf->GetBufferDeviceAddress();
    }

    constexpr size_t mlSize = sizeof(mModelData.meshes[0].meshlets[0]);
    constexpr size_t mlVertSize =
        sizeof(mModelData.meshes[0].meshletVertices[0]);
    constexpr size_t mlTriSize =
        sizeof(mModelData.meshes[0].meshletTriangles[0]);
    const size_t mlBufSize = mMeshletCount * mlSize;
    const size_t mlVertBufSize = mMeshletVertexCount * mlVertSize;
    const size_t mlTriBufSize = mMeshletTriangleCount * mlTriSize;

    // Meshlets buffers
    {
        mBuffers.mMeshletBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet").c_str(), mlBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletVertBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet vertices").c_str(), mlVertBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletTriBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Meshlet triangles").c_str(), mlTriBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);

        mBuffers.mMeshletBufAddr =
            mBuffers.mMeshletBuf->GetBufferDeviceAddress();
        mBuffers.mMeshletVertBufAddr =
            mBuffers.mMeshletVertBuf->GetBufferDeviceAddress();
        mBuffers.mMeshletTriBufAddr =
            mBuffers.mMeshletTriBuf->GetBufferDeviceAddress();
    }

    uint32_t meshCount = mModelData.meshes.size();

    // offsets data buffer *NO index*
    // vertex offsets + meshletoffsets + meshletVertices offsets + meshlettriangles offsets + meshlet counts
    const size_t offsetsBufSize = meshCount * 5 * sizeof(uint32_t);
    {
        mBuffers.mMeshDataBuf = context->CreateDeviceLocalBufferResource(
            (mName + " Offsets data").c_str(), offsetsBufSize,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        mBuffers.mMeshDataBufAddr =
            mBuffers.mMeshDataBuf->GetBufferDeviceAddress();
    }

    // copy through staging buffer
    {
        auto staging = context->CreateStagingBuffer(
            "", vpBufSize + vnBufSize + vtBufSize + mlBufSize + mlVertBufSize
                    + mlTriBufSize + offsetsBufSize);

        auto data = static_cast<char*>(staging->GetMapPtr());

        // vetices
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + mMeshDatas.vertexOffsets[i] * vpSize,
                   mModelData.meshes[i].vertices.positions.data(),
                   mModelData.meshes[i].vertices.positions.size() * vpSize);
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + mMeshDatas.vertexOffsets[i] * vnSize,
                   mModelData.meshes[i].vertices.normals.data(),
                   mModelData.meshes[i].vertices.normals.size() * vnSize);
        }

        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize
                       + mMeshDatas.vertexOffsets[i] * vtSize,
                   mModelData.meshes[i].vertices.uvs.data(),
                   mModelData.meshes[i].vertices.uvs.size() * vtSize);
        }

        // meshlets
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize
                       + mMeshDatas.meshletOffsets[i] * mlSize,
                   mModelData.meshes[i].meshlets.data(),
                   mModelData.meshes[i].meshlets.size() * mlSize);
        }
        // meshlet vertices
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize + mlBufSize
                       + mMeshDatas.meshletVerticesOffsets[i] * mlVertSize,
                   mModelData.meshes[i].meshletVertices.data(),
                   mModelData.meshes[i].meshletVertices.size() * mlVertSize);
        }
        //meshlet triangles
        for (uint32_t i = 0; i < meshCount; ++i) {
            memcpy(data + vpBufSize + vnBufSize + vtBufSize + mlBufSize
                       + mlVertBufSize
                       + mMeshDatas.meshletTrianglesOffsets[i] * mlTriSize,
                   mModelData.meshes[i].meshletTriangles.data(),
                   mModelData.meshes[i].meshletTriangles.size() * mlTriSize);
        }
        // offsets
        auto meshCountSize = meshCount * sizeof(meshCount);
        auto totalOffsets = vpBufSize + vnBufSize + vtBufSize + mlBufSize
                          + mlVertBufSize + mlTriBufSize;
        for (uint32_t i = 0; i < meshCount; ++i) {}
        {
            memcpy(data + totalOffsets, mMeshDatas.vertexOffsets.data(),
                   meshCountSize);

            memcpy(data + totalOffsets + meshCountSize,
                   mMeshDatas.meshletOffsets.data(), meshCountSize);

            memcpy(data + totalOffsets + meshCountSize * 2,
                   mMeshDatas.meshletVerticesOffsets.data(), meshCountSize);

            memcpy(data + totalOffsets + meshCountSize * 3,
                   mMeshDatas.meshletTrianglesOffsets.data(), meshCountSize);

            memcpy(data + totalOffsets + meshCountSize * 4,
                   mMeshDatas.meshletCounts.data(), meshCountSize);
        }

        {
            auto cmd = context->CreateCmdBufToBegin(
                context->GetQueue(QueueType::Transfer));
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

            vk::BufferCopy meshletVertCopy {};
            meshletVertCopy.setSize(mlVertBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + mlBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshletVertBuf->GetBufferHandle(),
                            meshletVertCopy);

            vk::BufferCopy meshletTriCopy {};
            meshletTriCopy.setSize(mlTriBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + mlBufSize
                              + mlVertBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshletTriBuf->GetBufferHandle(),
                            meshletTriCopy);

            vk::BufferCopy offsetsCopy {};
            offsetsCopy.setSize(offsetsBufSize)
                .setSrcOffset(vpBufSize + vnBufSize + vtBufSize + mlBufSize
                              + mlVertBufSize + mlTriBufSize);
            cmd->copyBuffer(staging->GetHandle(),
                            mBuffers.mMeshDataBuf->GetBufferHandle(),
                            offsetsCopy);
        }

        mMeshletConstants.mVPBufAddr = mBuffers.mVPBufAddr;
        mMeshletConstants.mVNBufAddr = mBuffers.mVNBufAddr;
        mMeshletConstants.mVTBufAddr = mBuffers.mVTBufAddr;
        mMeshletConstants.mMeshletBufAddr = mBuffers.mMeshletBufAddr;
        mMeshletConstants.mMeshletVertBufAddr = mBuffers.mMeshletVertBufAddr;
        mMeshletConstants.mMeshletTriBufAddr = mBuffers.mMeshletTriBufAddr;

        mMeshletConstants.mVertOffsetBufAddr = mBuffers.mMeshDataBufAddr;
        mMeshletConstants.mMeshletOffsetBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize;
        mMeshletConstants.mMeshletVertOffsetBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize * 2;
        mMeshletConstants.mMeshletTrioffsetBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize * 3;
        mMeshletConstants.mMeshletCountBufAddr =
            mBuffers.mMeshDataBufAddr + meshCountSize * 4;
    }

    // indirect mesh task command buffer
    {
        mMeshTaskIndirectCmdBuffer =
            context
                ->CreateIndirectCmdBuffer<vk::DrawMeshTasksIndirectCommandEXT>(
                    (mName + " Mesh Task Indirect Command").c_str(),
                    mMeshTaskIndirectCmds.size());
        auto bufSize = mMeshTaskIndirectCmdBuffer->GetSize();

        auto staging = context->CreateStagingBuffer("", bufSize);

        void* data = staging->GetMapPtr();
        memcpy(data, mMeshTaskIndirectCmds.data(), bufSize);

        {
            auto cmd = context->CreateCmdBufToBegin(
                context->GetQueue(QueueType::Transfer));
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

uint32_t Geometry::GetIndexCount() const {
    return mIndexCount;
}

uint32_t Geometry::GetTriangleCount() const {
    return mTriangleCount;
}

uint32_t Geometry::GetMeshletCount() const {
    return mMeshletCount;
}

uint32_t Geometry::GetMeshletVertexCount() const {
    return mMeshletVertexCount;
}

uint32_t Geometry::GetMeshletTriangleCount() const {
    return mMeshletTriangleCount;
}

std::span<uint32_t> Geometry::GetVertexOffsets() {
    return mMeshDatas.vertexOffsets;
}

std::span<uint32_t> Geometry::GetIndexOffsets() {
    return mMeshDatas.indexOffsets;
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

LegacyDrawPushConstants Geometry::GetIndexDrawPushConstants() const {
    return mIndexDrawConstants;
}

LegacyDrawPushConstants* Geometry::GetIndexDrawPushConstantsPtr() {
    return &mIndexDrawConstants;
}

Buffer* Geometry::GetIndirectCmdBuffer() const {
    return mIndirectCmdBuffer.get();
}

Buffer* Geometry::GetMeshTaskIndirectCmdBuffer() const {
    return mMeshTaskIndirectCmdBuffer.get();
}

void Geometry::LoadModel(const char* output, bool optimizeMesh,
                         bool buildMeshlet, bool optimizeMeshlet) {
    auto modelPath = mPath.string();
    auto cisdiModelPath = modelPath + CISDI_3DModel_Subfix_Str;
    if (::std::filesystem::exists(cisdiModelPath)) {
        mModelData = IntelliDesign_NS::ModelData::Load(cisdiModelPath.c_str());
    } else {
        mModelData = IntelliDesign_NS::ModelData::Convert(
            modelPath.c_str(), mFlipYZ, output, optimizeMesh, buildMeshlet,
            optimizeMeshlet);
    }
}

void Geometry::GenerateStats() {
    const auto meshCount = mModelData.meshes.size();
    mMeshDatas.vertexOffsets.reserve(meshCount);
    mMeshDatas.indexOffsets.reserve(meshCount);
    mMeshTaskIndirectCmds.reserve(meshCount);
    mIndirectCmds.reserve(meshCount);
    for (auto& mesh : mModelData.meshes) {
        vk::DrawIndirectCommand indirectCmd {};
        indirectCmd.setInstanceCount(1)
            .setVertexCount(mesh.indices.size())
            .setFirstVertex(mIndexCount);
        mIndirectCmds.push_back(indirectCmd);

        vk::DrawMeshTasksIndirectCommandEXT meshTasksIndirectCmd {};
        meshTasksIndirectCmd
            .setGroupCountX(
                (mesh.meshlets.size() + TASK_SHADER_INVOCATION_COUNT - 1)
                / TASK_SHADER_INVOCATION_COUNT)
            .setGroupCountY(1)
            .setGroupCountZ(1);
        mMeshTaskIndirectCmds.push_back(meshTasksIndirectCmd);

        mMeshDatas.vertexOffsets.push_back(mVertexCount);
        mVertexCount += mesh.vertices.positions.size();

        mMeshDatas.indexOffsets.push_back(mIndexCount);
        mIndexCount += mesh.indices.size();

        mMeshDatas.meshletOffsets.push_back(mMeshletCount);
        mMeshletCount += mesh.meshlets.size();
        mMeshDatas.meshletCounts.push_back(mesh.meshlets.size());

        mMeshDatas.meshletVerticesOffsets.push_back(mMeshletVertexCount);
        mMeshletVertexCount += mesh.meshletVertices.size();

        mMeshDatas.meshletTrianglesOffsets.push_back(mMeshletTriangleCount);
        mMeshletTriangleCount += mesh.meshletTriangles.size();
    }
    mTriangleCount = mIndexCount / 3;
}

}  // namespace IntelliDesign_NS::Vulkan::Core