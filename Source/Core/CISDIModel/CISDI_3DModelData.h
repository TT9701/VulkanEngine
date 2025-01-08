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
        uint64_t header {};
        Version version {};
        uint32_t nodeCount {0};
        uint32_t meshCount {0};
        uint32_t materialCount {0};
        bool buildMeshlet {false};
    };

    struct Mesh;
    struct Material;

    struct Node {
        Type_STLString name {};
        uint32_t meshIdx {~0ui32};
        uint32_t materialIdx {~0ui32};
        uint32_t parentIdx {~0ui32};

        uint32_t childCount {0};
        Type_STLVector<uint32_t> childrenIdx {};
    };

    struct Mesh {
        struct MeshHeader {
            uint32_t vertexCount {0};
            uint32_t indexCount {0};
            uint32_t meshletCount {0};
            uint32_t meshletVertexCount {0};
            uint32_t meshletTriangleCount {0};
        };

        struct Vertices {
            // Vertices(::std::pmr::memory_resource* pMemPool)
            //     : positions {::std::pmr::polymorphic_allocator {}} {}

            Type_STLVector<Float4> positions {};
            Type_STLVector<Float2> normals {};
            Type_STLVector<Float2> uvs {};
        };

        MeshHeader header {};
        Vertices vertices {};
        Type_STLVector<uint32_t> indices {};

        Type_STLVector<meshopt_Meshlet> meshlets {};
        Type_STLVector<uint32_t> meshletVertices {};
        Type_STLVector<uint8_t> meshletTriangles {};
    };

    struct Material {
        Type_STLString name {};
        Float3 ambient {};
        Float3 diffuse {};
        Float3 emissive {};
        float opacity {};
    };

    Header header {};

    Type_STLString name {};

    Type_STLVector<Node> nodes {};

    Type_STLVector<Mesh> meshes {};

    Type_STLVector<Material> materials {};
};

CISDI_MODEL_DATA_API CISDI_3DModel Convert(const char* path, bool flipYZ,
                                           const char* output = nullptr,
                                           bool optimizeMesh = true,
                                           bool buildMeshlet = true,
                                           bool optimizeMeshlet = true);

CISDI_MODEL_DATA_API CISDI_3DModel Load(const char* path);

}  // namespace IntelliDesign_NS::ModelData