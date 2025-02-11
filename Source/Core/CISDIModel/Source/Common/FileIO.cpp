#include "FileIO.h"

#include <cassert>
#include <fstream>

#include "CISDI_3DModelData.h"
#include "Common.h"

namespace IntelliDesign_NS::ModelData {

std::string ProcessOutputPath(const char* input, const char* output) {
    auto inPath = ::std::filesystem::path(input);
    ::std::filesystem::path outputPath;

    if (inPath.extension() == CISDI_3DModel_Subfix_Str) {
        return inPath.string();
    }

    if (!output) {
        outputPath = inPath.string().append(CISDI_3DModel_Subfix_Str);
    } else {
        outputPath = ::std::filesystem::canonical(output);
        if (::std::filesystem::is_directory(outputPath)) {
            outputPath = outputPath.wstring().append(L"/").append(
                inPath.wstring().append(CISDI_3DModel_Subfix_WStr));
        } else {
            throw ::std::runtime_error(
                "ERROR::CISDI_3DMODELDATA::CONVERT: Output is not a "
                "directory!");
        }
    }
    return outputPath.string();
}

void WriteDataHeader(std::ofstream& ofs, CISDI_3DModel::Header header) {
    ofs.write((char*)&header, sizeof(header));
}

void WriteString(std::ofstream& ofs, const char* str) {
    size_t nameLen = strlen(str);
    ofs.write((char*)&nameLen, sizeof(nameLen));
    ofs.write(str, nameLen);
}

void WriteNodeUserProperties(std::ofstream& ofs, CISDI_Node const& node) {
    assert(node.userPropertyCount == node.userProperties.size());
    ofs.write((char*)&node.userPropertyCount, sizeof(node.userPropertyCount));
    for (auto const& prop : node.userProperties) {
        WriteString(ofs, prop.first.c_str());

        ::std::underlying_type_t<UserPropertyValueTypeEnum> type =
            prop.second.index();
        ofs.write((char*)&type, sizeof(type));

        ::std::visit(
            [&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (::std::is_same_v<T, Type_STLString>) {
                    WriteString(ofs, v.c_str());
                } else {
                    ofs.write((char*)&v, sizeof(v));
                }
            },
            prop.second);
    }
}

void WriteNodes(std::ofstream& ofs, Type_STLVector<CISDI_Node> const& nodes) {
    for (auto const& node : nodes) {
        WriteString(ofs, node.name.c_str());
        auto offset = strlen(node.name.c_str());
        ofs.write((char*)&node.meshIdx, sizeof(uint32_t) * 4);
        if (node.childCount > 0)
            ofs.write((char*)node.childrenIdx.data(),
                      sizeof(node.childrenIdx[0]) * node.childCount);

        WriteNodeUserProperties(ofs, node);
    }
}

void WriteMeshHeader(std::ofstream& ofs,
                     CISDI_3DModel::CISDI_Mesh::MeshHeader const& meshHeader) {
    ofs.write((char*)&meshHeader, sizeof(meshHeader));
}

void WriteBoundingBox(std::ofstream& ofs, AABoundingBox const& box) {
    ofs.write((char*)&box, sizeof(box));
}

void WriteMeshes(std::ofstream& ofs,
                 Type_STLVector<CISDI_3DModel::CISDI_Mesh> const& meshes) {
    for (auto const& mesh : meshes) {
        WriteMeshHeader(ofs, mesh.header);

        // TODO: other attributes

        // meshlet datas
        if (mesh.header.meshletCount > 0) {
            auto const& infos =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>();
            ofs.write((char*)infos.data(),
                      sizeof(infos[0]) * mesh.header.meshletCount);
        }

        if (mesh.header.meshletTriangleCount > 0) {
            auto const& triangles =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Triangle>();
            ofs.write((char*)triangles.data(),
                      sizeof(triangles[0]) * mesh.header.meshletTriangleCount);
        }

        for (uint32_t i = 0; i < mesh.header.meshletCount; ++i) {
            auto const& vertices =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Vertex>()[i];

            auto const& vertCount =
                mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>()[i]
                    .vertexCount;

            auto const& vertPositions =
                vertices.GetProperty<VertexAttributeEnum::Position>();

            auto const& vertNormals =
                vertices.GetProperty<VertexAttributeEnum::Normal>();

            auto const& vertUVs =
                vertices.GetProperty<VertexAttributeEnum::UV>();

            ofs.write((char*)vertPositions.data(),
                      sizeof(vertPositions[0]) * vertCount);

            ofs.write((char*)vertNormals.data(),
                      sizeof(vertNormals[0]) * vertCount);

            ofs.write((char*)vertUVs.data(), sizeof(vertUVs[0]) * vertCount);
        }

        if (mesh.header.meshletCount > 0) {
            auto const& boundingBoxes =
                mesh.meshlets
                    .GetProperty<MeshletPropertyTypeEnum::BoundingBox>();
            ofs.write((char*)boundingBoxes.data(),
                      sizeof(boundingBoxes[0]) * mesh.header.meshletCount);
        }

        // bounding box
        WriteBoundingBox(ofs, mesh.boundingBox);
    }
}

void WriteMaterial(std::ofstream& ofs,
                   Type_STLVector<CISDI_Material> const& materials) {
    for (auto const& material : materials) {
        WriteString(ofs, material.name.c_str());
        ofs.write((char*)&material.data, sizeof(material.data));
    }
}

void Write_CISDI_File(const char* outputPath, CISDI_3DModel const& data) {
    ::std::ofstream out(outputPath, ::std::ios::out | ::std::ios::binary);

    if (!out.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("fail to open file: ") + outputPath).c_str());
    }

    WriteDataHeader(out, data.header);
    WriteString(out, data.name.c_str());
    WriteNodes(out, data.nodes);
    WriteMeshes(out, data.meshes);
    WriteMaterial(out, data.materials);
    WriteBoundingBox(out, data.boundingBox);
}

void ReadString(std::ifstream& in, Type_STLString& str) {
    size_t nameLen;
    in.read((char*)&nameLen, sizeof(nameLen));
    str.resize(nameLen);
    in.read(str.data(), nameLen);
}

void ReadDataHeader(std::ifstream& in, CISDI_3DModel::Header& header) {
    in.read((char*)&header, sizeof(header));
    if (CISDI_3DModel_HEADER_UINT64 != header.header) {
        throw ::std::runtime_error("ERROR::CISDI_3DModel::Convert");
    }
}

template <UserPropertyValueTypeEnum Enum>
using UserPropertyValueType =
    ::std::variant_alternative_t<static_cast<size_t>(Enum),
                                 Type_UserPropertyValue>;

void ReadNode(std::ifstream& in, uint32_t nodeCount,
              Type_STLVector<CISDI_Node>& nodes,
              std::pmr::memory_resource* pMemPool) {
    nodes.resize(nodeCount, CISDI_Node {pMemPool});
    for (auto& node : nodes) {
        ReadString(in, node.name);
        in.read((char*)&node.meshIdx, sizeof(uint32_t) * 4);
        if (node.childCount > 0) {
            node.childrenIdx.resize(node.childCount);
            in.read((char*)node.childrenIdx.data(),
                    sizeof(node.childrenIdx[0]) * node.childCount);
        }
        in.read((char*)&node.userPropertyCount, sizeof(node.userPropertyCount));
        for (uint32_t i = 0; i < node.userPropertyCount; ++i) {
            Type_STLString key;
            ReadString(in, key);

            UserPropertyValueTypeEnum type {};

            in.read((char*)&type, sizeof(type));

            switch (type) {
                case UserPropertyValueTypeEnum::Bool: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::Bool>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::Char: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::Char>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::UChar: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::UChar>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::Int: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::Int> value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::UInt: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::UInt>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::LongLong: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::LongLong>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::ULongLong: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::ULongLong>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::Float: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::Float>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::Double: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::Double>
                        value;
                    in.read((char*)&value, sizeof(value));
                    node.userProperties[key] = value;
                } break;
                case UserPropertyValueTypeEnum::String: {
                    UserPropertyValueType<UserPropertyValueTypeEnum::String>
                        value;
                    ReadString(in, value);
                    node.userProperties[key] = value;
                } break;
            }
        }
    }
}

void ReadBoundingBox(std::ifstream& in, AABoundingBox& box) {
    in.read((char*)&box, sizeof(box));
}

void ReadMesh(std::ifstream& in, uint32_t meshCount,
              Type_STLVector<CISDI_3DModel::CISDI_Mesh>& meshes,
              std::pmr::memory_resource* pMemPool) {
    meshes.resize(meshCount);
    for (auto& mesh : meshes) {
        in.read((char*)&mesh.header, sizeof(mesh.header));

        // TODO: other attributes
        auto& infos =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Info>();
        infos = Type_STLVector<MeshletInfo>(mesh.header.meshletCount, pMemPool);
        in.read((char*)infos.data(),
                sizeof(infos[0]) * mesh.header.meshletCount);

        auto& triangles =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Triangle>();
        triangles =
            Type_STLVector<uint8_t>(mesh.header.meshletTriangleCount, pMemPool);
        in.read((char*)triangles.data(),
                sizeof(triangles[0]) * mesh.header.meshletTriangleCount);

        auto& vertices =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::Vertex>();
        vertices =
            Type_STLVector<CISDI_Vertices>(mesh.header.meshletCount, pMemPool);
        for (uint32_t i = 0; i < mesh.header.meshletCount; ++i) {
            auto vertCount = infos[i].vertexCount;
            auto& vertPostions =
                vertices[i].GetProperty<VertexAttributeEnum::Position>();
            vertPostions = Type_STLVector<UInt16_3>(vertCount, pMemPool);
            in.read((char*)vertPostions.data(),
                    sizeof(vertPostions[0]) * vertCount);

            auto& vertNormals =
                vertices[i].GetProperty<VertexAttributeEnum::Normal>();
            vertNormals = Type_STLVector<Int16_2>(vertCount, pMemPool);
            in.read((char*)vertNormals.data(),
                    sizeof(vertNormals[0]) * vertCount);

            auto& vertUVs = vertices[i].GetProperty<VertexAttributeEnum::UV>();
            vertUVs = Type_STLVector<UInt16_2>(vertCount, pMemPool);
            in.read((char*)vertUVs.data(), sizeof(vertUVs[0]) * vertCount);
        }

        auto& boundingBoxes =
            mesh.meshlets.GetProperty<MeshletPropertyTypeEnum::BoundingBox>();
        boundingBoxes =
            Type_STLVector<AABoundingBox>(mesh.header.meshletCount, pMemPool);
        in.read((char*)boundingBoxes.data(),
                sizeof(boundingBoxes[0]) * mesh.header.meshletCount);

        ReadBoundingBox(in, mesh.boundingBox);
    }
}

void ReadMaterial(std::ifstream& in, uint32_t materialCount,
                  Type_STLVector<CISDI_Material>& materials,
                  std::pmr::memory_resource* pMemPool) {
    materials.resize(materialCount, CISDI_Material {pMemPool});
    for (auto& material : materials) {
        ReadString(in, material.name);
        in.read((char*)&material.data, sizeof(material.data));
    }
}

CISDI_3DModel Read_CISDI_File(const char* path,
                              std::pmr::memory_resource* pMemPool) {
    ::std::ifstream in(path, ::std::ios::binary);
    if (!in.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("fail to open file: ") + path).c_str());
    }

    CISDI_3DModel data {pMemPool};

    ReadDataHeader(in, data.header);

    // version check 
    if (CISDI_3DModel_VERSION != data.header.version) {
        throw ::std::runtime_error("ERROR::CISDI_3DModel::Convert");
    }

    ReadString(in, data.name);
    ReadNode(in, data.header.nodeCount, data.nodes, pMemPool);
    ReadMesh(in, data.header.meshCount, data.meshes, pMemPool);
    ReadMaterial(in, data.header.materialCount, data.materials, pMemPool);
    ReadBoundingBox(in, data.boundingBox);

    return data;
}

}  // namespace IntelliDesign_NS::ModelData