#pragma once

#include "Source/Common/Common.h"

#ifdef CISDI_MODEL_DATA_EXPORTS
#define CISDI_MODEL_DATA_API __declspec(dllexport)
#else
#define CISDI_MODEL_DATA_API __declspec(dllimport)
#endif

namespace IntelliDesign_NS::ModelData {

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;

constexpr Version CISDI_3DModel_VERSION = {0ui8, 4ui8, 1ui16};

// TODO: add pmr mempool param in ctor.

struct CISDI_3DModel {
    struct Header {
        uint64_t header {0};
        Version version {};
        uint32_t nodeCount {0};
        uint32_t meshCount {0};
        uint32_t materialCount {0};
    };

    struct Mesh;

    struct Node {
        Type_STLString name {};
        uint32_t meshIdx {~0ui32};
        uint32_t materialIdx {~0ui32};
        uint32_t parentIdx {~0ui32};

        uint32_t childCount {0};
        Type_STLVector<uint32_t> childrenIdx {};

        uint32_t userPropertyCount {0};
        Type_STLUnorderedMap_String<Type_UserPropertyValue> userProperties {};
    };

    struct Mesh {
        struct MeshHeader {
            uint32_t vertexCount {0};
            uint32_t meshletCount {0};
            uint32_t meshletTriangleCount {0};
        };

        MeshHeader header {};

        Meshlets meshlets {};

        AABoundingBox boundingBox {};
    };

    Header header {};

    Type_STLString name {};

    Type_STLVector<Node> nodes {};

    Type_STLVector<Mesh> meshes {};

    Type_STLVector<Material> materials {};

    AABoundingBox boundingBox {};
};

CISDI_MODEL_DATA_API CISDI_3DModel Convert(const char* path, bool flipYZ,
                                           const char* output = nullptr);

CISDI_MODEL_DATA_API CISDI_3DModel Load(const char* path);

}  // namespace IntelliDesign_NS::ModelData