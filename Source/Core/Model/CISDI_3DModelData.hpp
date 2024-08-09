#pragma once

#include <stdint.h>
#include <vector>

#define CISDI_3DModel_Subfix ".cisdi"

struct CISDI_3DModelDataVersion {
    uint8_t  major;
    uint8_t  minor;
    uint16_t patch;
};

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;
constexpr CISDI_3DModelDataVersion CISDI_3DModel_VERSION = {0ui8, 0ui8, 1ui16};

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
            struct Float3 {
                float x, y, z;
            };

            ::std::vector<Float3> positions;
            ::std::vector<Float3> normals;
            // TODO: other attributes
        } vertices;

        ::std::vector<uint32_t> indices;
    };

    ::std::vector<CISDI_Mesh> meshes;

    static void Convert(const char* path, bool flipYZ, const char* output = nullptr);

    static CISDI_3DModelData Load(const char* path);
};