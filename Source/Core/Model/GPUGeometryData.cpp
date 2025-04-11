#include "GPUGeometryData.h"

#include "Core/Application/Application.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

GPUGeometryData::GPUGeometryData(VulkanContext& context,
                                 ModelData::CISDI_3DModel const& model,
                                 uint32_t maxMeshCountPerDGCSequence)
    : mName(model.name),
      mSequenceCount((model.meshes.size() + maxMeshCountPerDGCSequence - 1)
                     / maxMeshCountPerDGCSequence) {
    mMeshDatas.resize(mSequenceCount);
    mMeshTaskIndirectCmds.resize(mSequenceCount);
    mBuffers.resize(mSequenceCount);
    mMeshletConstants.resize(mSequenceCount);
    mMeshTaskIndirectCmdBuffer.resize(mSequenceCount);

    GenerateStats(model, maxMeshCountPerDGCSequence);
    GenerateMeshletBuffers(context, model, maxMeshCountPerDGCSequence);
}

namespace {

size_t RoundUpTo256(size_t value) {
    return (value + 255) & ~255;
}

template <class T>
SharedPtr<Buffer> CreateStorageBuffer_WithData(
    VulkanContext& context, const char* name, uint32_t count, const T* data,
    vk::BufferUsageFlags usage = (vk::BufferUsageFlags)0) {
    size_t size = sizeof(T) * count;
    size_t roundupSize = RoundUpTo256(sizeof(T) * count);
    auto ptr = context.CreateStorageBuffer(
        name, roundupSize,
        usage | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    ptr->CopyData(data, size);
    return ptr;
}

template <ModelData::VertexAttributeEnum Enum>
auto ExtractVertexAttribute(ModelData::CISDI_3DModel const& modelData,
                            uint32_t meshOffset, uint32_t meshCount,
                            uint32_t vertCount) {
    using Type = ModelData::CISDI_Vertices::PropertyType<Enum>;

    Type tmp {};
    tmp.reserve(vertCount);
    for (uint32_t meshIdx = meshOffset; meshIdx < meshOffset + meshCount;
         ++meshIdx) {
        auto const& mesh = modelData.meshes[meshIdx];
        for (auto const& vertices :
             mesh.meshlets
                 .GetProperty<ModelData::MeshletPropertyTypeEnum::Vertex>()) {
            auto const& data = vertices.GetProperty<Enum>();
            tmp.insert(tmp.end(), data.begin(), data.end());
        }
    }
    return tmp;
}

template <ModelData::MeshletPropertyTypeEnum Enum>
auto ExtractMeshletProperty(ModelData::CISDI_3DModel const& modelData,
                            uint32_t meshOffset, uint32_t meshCount,
                            uint32_t count) {
    using Type = ModelData::CISDI_Meshlets::PropertyType<Enum>;

    Type tmp {};
    tmp.reserve(count);
    for (uint32_t meshIdx = meshOffset; meshIdx < meshOffset + meshCount;
         ++meshIdx) {
        auto const& mesh = modelData.meshes[meshIdx];
        auto const& data = mesh.meshlets.GetProperty<Enum>();
        tmp.insert(tmp.end(), data.begin(), data.end());
    }
    return tmp;
}

}  // namespace

void GPUGeometryData::GenerateMeshletBuffers(
    VulkanContext& context, ModelData::CISDI_3DModel const& model,
    uint32_t maxMeshCount) {
    // positions buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        // auto tmp =
        //     ExtractVertexAttribute<ModelData::VertexAttributeEnum::Position>(
        //         model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
        //         meshData.stats.mVertexCount);

        using Type = Type_STLVector<ModelData::UInt16_4>;
        Type tmp {};
        tmp.reserve(meshData.stats.mVertexCount);
        for (uint32_t meshIdx = seqIdx * maxMeshCount;
             meshIdx < seqIdx * maxMeshCount + meshData.stats.mMeshCount;
             ++meshIdx) {
            auto const& mesh = model.meshes[meshIdx];
            for (auto const& vertices :
                 mesh.meshlets.GetProperty<
                     ModelData::MeshletPropertyTypeEnum::Vertex>()) {
                auto const& data = vertices.GetProperty<
                    ModelData::VertexAttributeEnum::Position>();

                for (auto const& d : data) {
                    tmp.emplace_back(d.x, d.y, d.z, 0);
                }
            }
        }

        buffer.mVPBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Position").c_str(),
            meshData.stats.mVertexCount, tmp.data());

        mMeshletConstants[seqIdx].mVPBufAddr =
            buffer.mVPBuf->GetDeviceAddress();
    }

    // normals buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        auto tmp =
            ExtractVertexAttribute<ModelData::VertexAttributeEnum::Normal>(
                model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
                meshData.stats.mVertexCount);

        buffer.mVNBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Normal").c_str(),
            meshData.stats.mVertexCount, tmp.data());

        mMeshletConstants[seqIdx].mVNBufAddr =
            buffer.mVNBuf->GetDeviceAddress();
    }

    // texcoords buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        auto tmp = ExtractVertexAttribute<ModelData::VertexAttributeEnum::UV>(
            model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
            meshData.stats.mVertexCount);

        buffer.mVTBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Texcoords").c_str(),
            meshData.stats.mVertexCount, tmp.data());

        mMeshletConstants[seqIdx].mVTBufAddr =
            buffer.mVTBuf->GetDeviceAddress();
    }

    // meshlet infos buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        auto tmp =
            ExtractMeshletProperty<ModelData::MeshletPropertyTypeEnum::Info>(
                model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
                meshData.stats.mMeshletCount);

        buffer.mMeshletBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet").c_str(), meshData.stats.mMeshletCount,
            tmp.data());

        mMeshletConstants[seqIdx].mMeshletBufAddr =
            buffer.mMeshletBuf->GetDeviceAddress();
    }

    // meshlet triangles buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        auto tmp = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::Triangle>(
            model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
            meshData.stats.mMeshletTriangleCount);

        buffer.mMeshletTriBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet").c_str(),
            meshData.stats.mMeshletTriangleCount, tmp.data());

        mMeshletConstants[seqIdx].mMeshletTriBufAddr =
            buffer.mMeshletTriBuf->GetDeviceAddress();
    }

    // aabb data buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];
        auto& pc = mMeshletConstants[seqIdx];

        ::std::vector<ModelData::AABoundingBox> tmp;
        tmp.reserve(1 + meshData.stats.mMeshCount
                    + meshData.stats.mMeshletCount);

        tmp.push_back(model.boundingBox);
        for (uint32_t i = 0; i < meshData.stats.mMeshCount; ++i) {
            tmp.push_back(model.meshes[i].boundingBox);
        }

        auto meshletBB = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::BoundingBox>(
            model, seqIdx * maxMeshCount, meshData.stats.mMeshCount,
            meshData.stats.mMeshCount);
        tmp.insert(tmp.end(), meshletBB.begin(), meshletBB.end());

        buffer.mBoundingBoxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Bounding Box").c_str(), tmp.size(), tmp.data());

        pc.mBoundingBoxBufAddr = buffer.mBoundingBoxBuf->GetDeviceAddress();
        pc.mMeshletBoundingBoxBufAddr = pc.mBoundingBoxBufAddr
                                      + sizeof(ModelData::AABoundingBox)
                                            * (meshData.stats.mMeshCount + 1);
    }

    // material data buffer
    {
        ::std::vector<ModelData::CISDI_Material::Data> datas {};
        datas.reserve(model.materials.size());
        for (auto const& material : model.materials) {
            datas.push_back(material.data);
        }

        for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
            auto& buffer = mBuffers[seqIdx];
            buffer.mMaterialBuf = CreateStorageBuffer_WithData(
                context, (mName + " Materials").c_str(), datas.size(),
                datas.data());

            mMeshletConstants[seqIdx].mMaterialBufAddr =
                buffer.mMaterialBuf->GetDeviceAddress();
        }
    }

    // material indices buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];

        ::std::vector<uint32_t> meshMaterialIndices(meshData.stats.mMeshCount);
        for (auto const& node : model.nodes) {
            if (node.meshIdx == -1)
                continue;

            if (node.meshIdx < seqIdx * maxMeshCount
                || node.meshIdx
                       >= seqIdx * maxMeshCount + meshData.stats.mMeshCount)
                continue;

            meshMaterialIndices[node.meshIdx - seqIdx * maxMeshCount] =
                node.materialIdx;
        }
        buffer.mMeshMaterialIdxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Mesh Material Indices").c_str(),
            meshData.stats.mMeshCount, meshMaterialIndices.data());

        mMeshletConstants[seqIdx].mMeshMaterialIdxBufAddr =
            buffer.mMeshMaterialIdxBuf->GetDeviceAddress();
    }

    // offsets buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto const& meshData = mMeshDatas[seqIdx];
        auto& buffer = mBuffers[seqIdx];
        auto& pc = mMeshletConstants[seqIdx];

        ::std::vector<uint32_t> tmp {};
        tmp.reserve(meshData.stats.mMeshCount * 4);
        tmp.insert(tmp.end(), meshData.vertexOffsets.begin(),
                   meshData.vertexOffsets.end());
        tmp.insert(tmp.end(), meshData.meshletOffsets.begin(),
                   meshData.meshletOffsets.end());
        tmp.insert(tmp.end(), meshData.meshletTrianglesOffsets.begin(),
                   meshData.meshletTrianglesOffsets.end());
        tmp.insert(tmp.end(), meshData.meshletCounts.begin(),
                   meshData.meshletCounts.end());

        buffer.mMeshDataBuf = CreateStorageBuffer_WithData(
            context, (mName + " Offsets data").c_str(),
            meshData.stats.mMeshCount * 4, tmp.data());

        uint32_t meshCountSize = meshData.stats.mMeshCount * sizeof(uint32_t);

        pc.mVertOffsetBufAddr = buffer.mMeshDataBuf->GetDeviceAddress();
        pc.mMeshletOffsetBufAddr = pc.mVertOffsetBufAddr + meshCountSize;
        pc.mMeshletTrioffsetBufAddr = pc.mVertOffsetBufAddr + meshCountSize * 2;
        pc.mMeshletCountBufAddr = pc.mVertOffsetBufAddr + meshCountSize * 3;
    }

    // stats buffer
    {
        GeoStatistics stats {};

        for (auto const& data : mMeshDatas) {
            stats.vertexCount += data.stats.mVertexCount;
            stats.meshletCount += data.stats.mMeshletCount;
            stats.triangleCount += data.stats.mMeshletTriangleCount;
            stats.materialCount += model.materials.size();
        }

        for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
            auto& buffer = mBuffers[seqIdx];
            auto& pc = mMeshletConstants[seqIdx];

            buffer.mStatsBuffer = CreateStorageBuffer_WithData(
                context, (mName + "stats").c_str(), 1, &stats);

            pc.mStatsBufferAddr = buffer.mStatsBuffer->GetDeviceAddress();
        }
    }

    // indirect mesh task command buffer
    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto& cmdBuffer = mMeshTaskIndirectCmdBuffer[seqIdx];
        cmdBuffer =
            context
                .CreateIndirectCmdBuffer<vk::DrawMeshTasksIndirectCommandEXT>(
                    (mName + " Mesh Task Indirect Command").c_str(),
                    mMeshTaskIndirectCmds[seqIdx].size());

        cmdBuffer->CopyData(mMeshTaskIndirectCmds[seqIdx].data(),
                            cmdBuffer->GetSize());
    }
}

Type_STLString const& GPUGeometryData::GetName() const {
    return mName;
}

MeshletPushConstants GPUGeometryData::GetMeshletPushContants(
    uint32_t idx) const {
    return mMeshletConstants[idx];
}

vk::DrawIndirectCountIndirectCommandEXT
GPUGeometryData::GetDrawIndirectCmdBufInfo(uint32_t idx) const {
    auto const& buffer = *mMeshTaskIndirectCmdBuffer[idx];
    return {buffer.GetDeviceAddress(), (uint32_t)buffer.GetStride(),
            buffer.GetCount()};
}

uint32_t GPUGeometryData::GetSequenceCount() const {
    return mSequenceCount;
}

GPUGeometryData::MeshDatas::Stats GPUGeometryData::GetStats() const {
    MeshDatas::Stats stats {};
    for (auto const& data : mMeshDatas) {
        stats.mVertexCount += data.stats.mVertexCount;
        stats.mMeshCount += data.stats.mMeshCount;
        stats.mMeshletCount += data.stats.mMeshletCount;
        stats.mMeshletTriangleCount += data.stats.mMeshletTriangleCount;
    }

    return stats;
}

void GPUGeometryData::GenerateStats(ModelData::CISDI_3DModel const& model,
                                    uint32_t maxMeshCount) {
    size_t meshCount = model.meshes.size();

    for (auto& data : mMeshDatas) {
        if (meshCount > maxMeshCount) {
            data.stats.mMeshCount = maxMeshCount;
            meshCount -= maxMeshCount;
        } else {
            data.stats.mMeshCount = meshCount;
        }
    }

    for (uint32_t seqIdx = 0; seqIdx < mSequenceCount; ++seqIdx) {
        auto& meshData = mMeshDatas[seqIdx];
        auto& indirectCmds = mMeshTaskIndirectCmds[seqIdx];

        meshData.vertexOffsets.reserve(meshData.stats.mMeshCount);
        indirectCmds.reserve(meshData.stats.mMeshCount);

        for (uint32_t meshIdx = 0; meshIdx < meshData.stats.mMeshCount;
             ++meshIdx) {
            auto& mesh = model.meshes[seqIdx * maxMeshCount + meshIdx];

            vk::DrawMeshTasksIndirectCommandEXT meshTasksIndirectCmd {};
            const size_t infoSize =
                mesh.meshlets
                    .GetProperty<ModelData::MeshletPropertyTypeEnum::Info>()
                    .size();

            meshTasksIndirectCmd
                .setGroupCountX((infoSize + TASK_SHADER_INVOCATION_COUNT - 1)
                                / TASK_SHADER_INVOCATION_COUNT)
                .setGroupCountY(1)
                .setGroupCountZ(1);
            indirectCmds.push_back(meshTasksIndirectCmd);

            meshData.vertexOffsets.push_back(meshData.stats.mVertexCount);
            meshData.stats.mVertexCount += mesh.header.vertexCount;

            meshData.meshletOffsets.push_back(meshData.stats.mMeshletCount);
            meshData.stats.mMeshletCount += infoSize;
            meshData.meshletCounts.push_back(infoSize);

            auto const& triangles = mesh.meshlets.GetProperty<
                ModelData::MeshletPropertyTypeEnum::Triangle>();
            meshData.meshletTrianglesOffsets.push_back(
                meshData.stats.mMeshletTriangleCount);
            meshData.stats.mMeshletTriangleCount += triangles.size();
        }
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core