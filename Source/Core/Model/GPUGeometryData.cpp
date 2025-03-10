#include "GPUGeometryData.h"

#include "Core/Application/Application.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

GPUGeometryData::GPUGeometryData(VulkanContext& context,
                                 ModelData::CISDI_3DModel const& model)
    : mName(model.name) {
    GenerateStats(model);
    GenerateMeshletBuffers(context, model);
}

namespace {

template <class T>
SharedPtr<Buffer> CreateStorageBuffer_WithData(
    VulkanContext& context, const char* name, uint32_t count, const T* data,
    vk::BufferUsageFlags usage = (vk::BufferUsageFlags)0) {
    size_t size = sizeof(T) * count;
    auto ptr = context.CreateStorageBuffer(
        name, size,
        usage | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    ptr->CopyData(data, size);
    return ptr;
}

template <ModelData::VertexAttributeEnum Enum>
auto ExtractVertexAttribute(ModelData::CISDI_3DModel const& modelData,
                            uint32_t vertCount) {
    using Type = ModelData::CISDI_Vertices::PropertyType<Enum>;

    Type tmp {};
    tmp.reserve(vertCount);
    for (auto const& mesh : modelData.meshes) {
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
                            uint32_t count) {
    using Type = ModelData::CISDI_Meshlets::PropertyType<Enum>;

    Type tmp {};
    tmp.reserve(count);
    for (auto const& mesh : modelData.meshes) {
        auto const& data = mesh.meshlets.GetProperty<Enum>();
        tmp.insert(tmp.end(), data.begin(), data.end());
    }
    return tmp;
}

}  // namespace

void GPUGeometryData::GenerateMeshletBuffers(
    VulkanContext& context, ModelData::CISDI_3DModel const& model) {
    // positions buffer
    {
        auto tmp =
            ExtractVertexAttribute<ModelData::VertexAttributeEnum::Position>(
                model, mVertexCount);

        mBuffers.mVPBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Position").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVPBufAddr = mBuffers.mVPBuf->GetDeviceAddress();
    }

    // normals buffer
    {
        auto tmp =
            ExtractVertexAttribute<ModelData::VertexAttributeEnum::Normal>(
                model, mVertexCount);

        mBuffers.mVNBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Normal").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVNBufAddr = mBuffers.mVNBuf->GetDeviceAddress();
    }

    // texcoords buffer
    {
        auto tmp = ExtractVertexAttribute<ModelData::VertexAttributeEnum::UV>(
            model, mVertexCount);

        mBuffers.mVTBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Texcoords").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVTBufAddr = mBuffers.mVTBuf->GetDeviceAddress();
    }

    // meshlet infos buffer
    {
        auto tmp =
            ExtractMeshletProperty<ModelData::MeshletPropertyTypeEnum::Info>(
                model, mMeshletCount);

        mBuffers.mMeshletBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet").c_str(), mMeshletCount, tmp.data());

        mMeshletConstants.mMeshletBufAddr =
            mBuffers.mMeshletBuf->GetDeviceAddress();
    }

    // meshlet triangles buffer
    {
        auto tmp = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::Triangle>(
            model, mMeshletTriangleCount);

        mBuffers.mMeshletTriBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet triangles").c_str(),
            mMeshletTriangleCount, tmp.data());

        mMeshletConstants.mMeshletTriBufAddr =
            mBuffers.mMeshletTriBuf->GetDeviceAddress();
    }

    // aabb data buffer
    {
        mMeshCount = model.meshes.size();

        ::std::vector<ModelData::AABoundingBox> tmp;
        tmp.reserve(1 + mMeshCount + mMeshletCount);

        tmp.push_back(model.boundingBox);
        for (uint32_t i = 0; i < mMeshCount; ++i) {
            tmp.push_back(model.meshes[i].boundingBox);
        }

        auto meshletBB = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::BoundingBox>(model, mMeshCount);
        tmp.insert(tmp.end(), meshletBB.begin(), meshletBB.end());

        mBuffers.mBoundingBoxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Bounding Box").c_str(), tmp.size(), tmp.data());

        mMeshletConstants.mBoundingBoxBufAddr =
            mBuffers.mBoundingBoxBuf->GetDeviceAddress();
        mMeshletConstants.mMeshletBoundingBoxBufAddr =
            mMeshletConstants.mBoundingBoxBufAddr
            + sizeof(ModelData::AABoundingBox) * (mMeshCount + 1);
    }

    // material data buffer
    {
        ::std::vector<ModelData::CISDI_Material::Data> datas {};
        datas.reserve(model.materials.size());
        for (auto const& material : model.materials) {
            datas.push_back(material.data);
        }

        mBuffers.mMaterialBuf = CreateStorageBuffer_WithData(
            context, (mName + " Materials").c_str(), datas.size(),
            datas.data());

        mMeshletConstants.mMaterialBufAddr =
            mBuffers.mMaterialBuf->GetDeviceAddress();
    }

    // material indices buffer
    {
        ::std::vector<uint32_t> meshMaterialIndices(mMeshCount);
        for (auto const& node : model.nodes) {
            if (node.meshIdx == -1)
                continue;
            meshMaterialIndices[node.meshIdx] = node.materialIdx;
        }
        mBuffers.mMeshMaterialIdxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Mesh Material Indices").c_str(), mMeshCount,
            meshMaterialIndices.data());

        mMeshletConstants.mMeshMaterialIdxBufAddr =
            mBuffers.mMeshMaterialIdxBuf->GetDeviceAddress();
    }

    // offsets buffer
    {
        ::std::vector<uint32_t> tmp {};
        tmp.reserve(model.meshes.size() * 4);
        tmp.insert(tmp.end(), mMeshDatas.vertexOffsets.begin(),
                   mMeshDatas.vertexOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletOffsets.begin(),
                   mMeshDatas.meshletOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletTrianglesOffsets.begin(),
                   mMeshDatas.meshletTrianglesOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletCounts.begin(),
                   mMeshDatas.meshletCounts.end());

        mBuffers.mMeshDataBuf = CreateStorageBuffer_WithData(
            context, (mName + " Offsets data").c_str(), model.meshes.size() * 4,
            tmp.data());

        uint32_t meshCountSize = mMeshCount * sizeof(uint32_t);

        mMeshletConstants.mVertOffsetBufAddr =
            mBuffers.mMeshDataBuf->GetDeviceAddress();
        mMeshletConstants.mMeshletOffsetBufAddr =
            mMeshletConstants.mVertOffsetBufAddr + meshCountSize;
        mMeshletConstants.mMeshletTrioffsetBufAddr =
            mMeshletConstants.mVertOffsetBufAddr + meshCountSize * 2;
        mMeshletConstants.mMeshletCountBufAddr =
            mMeshletConstants.mVertOffsetBufAddr + meshCountSize * 3;
    }

    // indirect mesh task command buffer
    {
        mMeshTaskIndirectCmdBuffer =
            context
                .CreateIndirectCmdBuffer<vk::DrawMeshTasksIndirectCommandEXT>(
                    (mName + " Mesh Task Indirect Command").c_str(),
                    mMeshTaskIndirectCmds.size());

        mMeshTaskIndirectCmdBuffer->CopyData(
            mMeshTaskIndirectCmds.data(),
            mMeshTaskIndirectCmdBuffer->GetSize());
    }
}

Type_STLString const& GPUGeometryData::GetName() const {
    return mName;
}

uint32_t GPUGeometryData::GetMeshCount() const {
    return mMeshCount;
}

uint32_t GPUGeometryData::GetVertexCount() const {
    return mVertexCount;
}

uint32_t GPUGeometryData::GetMeshletCount() const {
    return mMeshletCount;
}

uint32_t GPUGeometryData::GetMeshletTriangleCount() const {
    return mMeshletTriangleCount;
}

std::span<uint32_t> GPUGeometryData::GetVertexOffsets() {
    return mMeshDatas.vertexOffsets;
}

GPUMeshBuffers& GPUGeometryData::GetMeshBuffer() {
    return mBuffers;
}

MeshletPushConstants GPUGeometryData::GetMeshletPushContants() const {
    return mMeshletConstants;
}

MeshletPushConstants* GPUGeometryData::GetMeshletPushContantsPtr() {
    return &mMeshletConstants;
}

Buffer* GPUGeometryData::GetMeshTaskIndirectCmdBuffer() const {
    return mMeshTaskIndirectCmdBuffer.get();
}

void GPUGeometryData::GenerateStats(ModelData::CISDI_3DModel const& model) {
    const auto meshCount = model.meshes.size();
    mMeshDatas.vertexOffsets.reserve(meshCount);
    mMeshTaskIndirectCmds.reserve(meshCount);
    for (auto& mesh : model.meshes) {
        vk::DrawMeshTasksIndirectCommandEXT meshTasksIndirectCmd {};
        auto const& infos =
            mesh.meshlets
                .GetProperty<ModelData::MeshletPropertyTypeEnum::Info>();

        meshTasksIndirectCmd
            .setGroupCountX((infos.size() + TASK_SHADER_INVOCATION_COUNT - 1)
                            / TASK_SHADER_INVOCATION_COUNT)
            .setGroupCountY(1)
            .setGroupCountZ(1);
        mMeshTaskIndirectCmds.push_back(meshTasksIndirectCmd);

        mMeshDatas.vertexOffsets.push_back(mVertexCount);
        mVertexCount += mesh.header.vertexCount;

        mMeshDatas.meshletOffsets.push_back(mMeshletCount);
        mMeshletCount += infos.size();
        mMeshDatas.meshletCounts.push_back(infos.size());

        auto const& triangles =
            mesh.meshlets
                .GetProperty<ModelData::MeshletPropertyTypeEnum::Triangle>();
        mMeshDatas.meshletTrianglesOffsets.push_back(mMeshletTriangleCount);
        mMeshletTriangleCount += triangles.size();
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core