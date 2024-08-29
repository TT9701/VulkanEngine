#pragma once

#include <stdint.h>
#include <vector>

#include <meshoptimizer.h>

#define CISDI_3DModel_Subfix ".cisdi"

template <class T>
using Vector = ::std::pmr::vector<T>;
using String = ::std::pmr::string;

struct CISDI_3DModelDataVersion {
    uint8_t  major;
    uint8_t  minor;
    uint16_t patch;
};

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;
constexpr CISDI_3DModelDataVersion CISDI_3DModel_VERSION = {0ui8, 1ui8, 1ui16};

struct CISDI_3DModelData {
    struct Header {
        uint64_t                 header;
        CISDI_3DModelDataVersion version;
        uint32_t                 meshCount;
    } header;

    struct CISDI_Mesh {
        struct MeshHeader {
            uint32_t vertexCount;
            uint32_t indexCount;
        } header;

        struct Vertices {
            struct Float2 {
                float x, y;
            };

            struct Float3 {
                float x, y, z;
            };

            Vector<Float3> positions;
            Vector<Float3> normals;
            // Vector<Float2> uvs;
            // Vector<Float3> tangents;
            // Vector<Float3> bitangents;

            // Vector<meshopt_Meshlet> meshlets;
            // Vector<uint32_t> meshletVertices;
            // Vector<uint8_t> meshletTriangles;
        } vertices;

        Vector<uint32_t> indices;
    };

    Vector<CISDI_Mesh> meshes;

    static void Convert(const char* path, bool flipYZ, const char* output = nullptr);

    static CISDI_3DModelData Load(const char* path);
};