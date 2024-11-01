#pragma once

#include <stdint.h>
#include <string>
#include <vector>

#include <meshoptimizer.h>

#define CISDI_3DModel_Subfix_Str ".cisdi"
#define CISDI_3DModel_Subfix_WStr L".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

template <class T>
using Type_STLVector = ::std::pmr::vector<T>;
using Type_STLString = ::std::string;

struct Version {
    uint8_t major;
    uint8_t minor;
    uint16_t patch;
};

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;
constexpr Version CISDI_3DModel_VERSION = {0ui8, 1ui8, 1ui16};

template <class T, uint32_t Dim>
struct Vec {
    T elem[Dim];

    T& operator[](uint32_t idx) { return elem[idx]; }
};

using Float2 = Vec<float, 2>;
using Float3 = Vec<float, 3>;
using Float4 = Vec<float, 4>;

struct CISDI_3DModel {
    struct Header {
        uint64_t header;
        Version version;
        uint32_t meshCount {0};
        bool buildMeshlet;
    } header;

    struct Mesh {
        struct MeshHeader {
            uint32_t vertexCount {0};
            uint32_t indexCount {0};
            uint32_t meshletCount {0};
            uint32_t meshletVertexCount {0};
            uint32_t meshletTriangleCount {0};
        } header;

        struct Vertices {
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

    static CISDI_3DModel Convert(const char* path, bool flipYZ,
                        const char* output = nullptr, bool optimizeMesh = true,
                        bool buildMeshlet = true, bool optimizeMeshlet = true);

    static CISDI_3DModel Load(const char* path);
};

}  // namespace IntelliDesign_NS::ModelData