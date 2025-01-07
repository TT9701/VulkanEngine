#pragma once

#include <meshoptimizer.h>

#include "Common.h"

#ifdef CISDI_MODEL_DATA_EXPORTS
#define CISDI_MODEL_DATA_API __declspec(dllexport)
#else
#define CISDI_MODEL_DATA_API __declspec(dllimport)
#endif

namespace IntelliDesign_NS::ModelData {

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;
constexpr Version CISDI_3DModel_VERSION = {0ui8, 1ui8, 1ui16};

struct CISDI_3DModel {
    struct Header {
        uint64_t header;
        Version version;
        uint32_t meshCount {0};
        bool buildMeshlet;
    } header;

    struct Mesh;
    struct Material;

    struct Node {
        Mesh* mesh;
        Material* material;
    };

    struct Mesh {
        struct MeshHeader {
            uint32_t vertexCount {0};
            uint32_t indexCount {0};
            uint32_t meshletCount {0};
            uint32_t meshletVertexCount {0};
            uint32_t meshletTriangleCount {0};
        } header;

        struct Vertices {
            // Vertices(::std::pmr::memory_resource* pMemPool)
            //     : positions {::std::pmr::polymorphic_allocator {}} {}

            Type_STLVector<Float4> positions;
            Type_STLVector<Float2> normals;
            Type_STLVector<Float2> uvs;
        } vertices;

        Type_STLVector<uint32_t> indices;

        Type_STLVector<meshopt_Meshlet> meshlets;
        Type_STLVector<uint32_t> meshletVertices;
        Type_STLVector<uint8_t> meshletTriangles;
    };

    Type_STLVector<Mesh> meshes;

    static CISDI_MODEL_DATA_API CISDI_3DModel
    Convert(const char* path, bool flipYZ, const char* output = nullptr,
            bool optimizeMesh = true, bool buildMeshlet = true,
            bool optimizeMeshlet = true);

    static CISDI_MODEL_DATA_API CISDI_3DModel Load(const char* path);
};

}  // namespace IntelliDesign_NS::ModelData