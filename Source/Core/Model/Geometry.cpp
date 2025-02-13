#include "Geometry.h"

#include "CISDI_3DModelData.h"
#include "Core/Application/Application.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

Geometry::Geometry(const char* path, ::std::pmr::memory_resource* pMemPool,
                   bool flipYZ, const char* output)
    : mFlipYZ(flipYZ),
      mPath(path),
      mDirectory(::std::filesystem::path {mPath}.remove_filename()),
      mName(mPath.stem().generic_string().c_str()),
      mModelData(LoadModel(output, pMemPool)) {
    GenerateStats();
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

void Geometry::GenerateMeshletBuffers(VulkanContext& context) {
    // positions buffer
    {
        auto tmp =
            ExtractVertexAttribute<ModelData::VertexAttributeEnum::Position>(
                mModelData, mVertexCount);

        mBuffers.mVPBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Position").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVPBufAddr = mBuffers.mVPBuf->GetDeviceAddress();
    }

    // normals buffer
    {
        auto tmp =
            ExtractVertexAttribute<ModelData::VertexAttributeEnum::Normal>(
                mModelData, mVertexCount);

        mBuffers.mVNBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Normal").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVNBufAddr = mBuffers.mVNBuf->GetDeviceAddress();
    }

    // texcoords buffer
    {
        auto tmp = ExtractVertexAttribute<ModelData::VertexAttributeEnum::UV>(
            mModelData, mVertexCount);

        mBuffers.mVTBuf = CreateStorageBuffer_WithData(
            context, (mName + " Vertex Texcoords").c_str(), mVertexCount,
            tmp.data());

        mMeshletConstants.mVTBufAddr = mBuffers.mVTBuf->GetDeviceAddress();
    }

    // meshlet infos buffer
    {
        auto tmp =
            ExtractMeshletProperty<ModelData::MeshletPropertyTypeEnum::Info>(
                mModelData, mMeshletCount);

        mBuffers.mMeshletBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet").c_str(), mMeshletCount, tmp.data());

        mMeshletConstants.mMeshletBufAddr =
            mBuffers.mMeshletBuf->GetDeviceAddress();
    }

    // meshlet triangles buffer
    {
        auto tmp = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::Triangle>(
            mModelData, mMeshletTriangleCount);

        mBuffers.mMeshletTriBuf = CreateStorageBuffer_WithData(
            context, (mName + " Meshlet triangles").c_str(),
            mMeshletTriangleCount, tmp.data());

        mMeshletConstants.mMeshletTriBufAddr =
            mBuffers.mMeshletTriBuf->GetDeviceAddress();
    }

    // aabb data buffer
    {
        uint32_t meshCount = mModelData.meshes.size();

        ::std::vector<ModelData::AABoundingBox> tmp;
        tmp.reserve(1 + meshCount + mMeshletCount);

        tmp.push_back(mModelData.boundingBox);
        for (uint32_t i = 0; i < meshCount; ++i) {
            tmp.push_back(mModelData.meshes[i].boundingBox);
        }

        auto meshletBB = ExtractMeshletProperty<
            ModelData::MeshletPropertyTypeEnum::BoundingBox>(mModelData,
                                                             meshCount);
        tmp.insert(tmp.end(), meshletBB.begin(), meshletBB.end());

        mBuffers.mBoundingBoxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Bounding Box").c_str(), tmp.size(), tmp.data());

        mMeshletConstants.mBoundingBoxBufAddr =
            mBuffers.mBoundingBoxBuf->GetDeviceAddress();
        mMeshletConstants.mMeshletBoundingBoxBufAddr =
            mMeshletConstants.mBoundingBoxBufAddr
            + sizeof(ModelData::AABoundingBox) * (meshCount + 1);
    }

    // material data buffer
    {
        ::std::vector<ModelData::CISDI_Material::Data> datas {};
        datas.reserve(mModelData.materials.size());
        for (auto const& material : mModelData.materials) {
            datas.push_back(material.data);
        }

        mBuffers.mMaterialBuf = CreateStorageBuffer_WithData(
            context, (mName + " Materials").c_str(), datas.size(),
            datas.data());

        mFragmentConstants.mMaterialBufAddr =
            mBuffers.mMaterialBuf->GetDeviceAddress();
    }

    // material indices buffer
    {
        uint32_t meshCount = mModelData.meshes.size();
        ::std::vector<uint32_t> meshMaterialIndices(meshCount);
        for (auto const& node : mModelData.nodes) {
            if (node.meshIdx == -1)
                continue;
            meshMaterialIndices[node.meshIdx] = node.materialIdx;
        }
        mBuffers.mMeshMaterialIdxBuf = CreateStorageBuffer_WithData(
            context, (mName + " Mesh Material Indices").c_str(), meshCount,
            meshMaterialIndices.data());

        mFragmentConstants.mMeshMaterialIdxBufAddr =
            mBuffers.mMeshMaterialIdxBuf->GetDeviceAddress();
    }

    // offsets buffer
    {
        ::std::vector<uint32_t> tmp {};
        tmp.reserve(mModelData.meshes.size() * 4);
        tmp.insert(tmp.end(), mMeshDatas.vertexOffsets.begin(),
                   mMeshDatas.vertexOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletOffsets.begin(),
                   mMeshDatas.meshletOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletTrianglesOffsets.begin(),
                   mMeshDatas.meshletTrianglesOffsets.end());
        tmp.insert(tmp.end(), mMeshDatas.meshletCounts.begin(),
                   mMeshDatas.meshletCounts.end());

        mBuffers.mMeshDataBuf = CreateStorageBuffer_WithData(
            context, (mName + " Offsets data").c_str(),
            mModelData.meshes.size() * 4, tmp.data());

        uint32_t meshCountSize = mModelData.meshes.size() * sizeof(uint32_t);

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

ModelData::CISDI_3DModel Geometry::LoadModel(
    const char* output, ::std::pmr::memory_resource* pMemPool) {
    auto modelPath = mPath.string();

    if (mPath.extension() == CISDI_3DModel_Subfix_Str) {
        return IntelliDesign_NS::ModelData::Load(mPath.string().c_str(),
                                                 pMemPool);
    }

    // auto cisdiModelPath = modelPath + CISDI_3DModel_Subfix_Str;
    //
    // if (::std::filesystem::exists(cisdiModelPath)) {
    //     return IntelliDesign_NS::ModelData::Load(cisdiModelPath.c_str(),
    //                                              pMemPool);
    // }

    return IntelliDesign_NS::ModelData::Convert(modelPath.c_str(), mFlipYZ,
                                                pMemPool, output);
}

void Geometry::GenerateStats() {
    const auto meshCount = mModelData.meshes.size();
    mMeshDatas.vertexOffsets.reserve(meshCount);
    mMeshTaskIndirectCmds.reserve(meshCount);
    for (auto& mesh : mModelData.meshes) {
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