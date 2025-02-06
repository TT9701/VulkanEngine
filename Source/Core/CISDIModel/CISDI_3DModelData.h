#pragma once

#include "Source/Common/Common.h"

#ifdef CISDI_MODEL_DATA_EXPORTS
#define CISDI_MODEL_DATA_API __declspec(dllexport)
#else
#define CISDI_MODEL_DATA_API __declspec(dllimport)
#endif

constexpr bool bUseCombinedImport = true;

namespace IntelliDesign_NS::ModelData {

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;

constexpr Version CISDI_3DModel_VERSION = {0ui8, 4ui8, 1ui16};

struct CISDI_3DModel {
    CISDI_3DModel(::std::pmr::memory_resource* pMemPool);

    struct Header {
        uint64_t header {0};
        Version version {};
        uint32_t nodeCount {0};
        uint32_t meshCount {0};
        uint32_t materialCount {0};
    };

    struct CISDI_Mesh {
        struct MeshHeader {
            uint32_t vertexCount {0};
            uint32_t meshletCount {0};
            uint32_t meshletTriangleCount {0};
        };

        MeshHeader header {};

        CISDI_Meshlets meshlets {};

        AABoundingBox boundingBox {};
    };

    Header header {};

    Type_STLString name;

    Type_STLVector<CISDI_Node> nodes;

    Type_STLVector<CISDI_Mesh> meshes;

    Type_STLVector<CISDI_Material> materials;

    AABoundingBox boundingBox {};
};

CISDI_MODEL_DATA_API CISDI_3DModel
Convert(const char* path, bool flipYZ, ::std::pmr::memory_resource* pMemPool,
        const char* output = nullptr);

CISDI_MODEL_DATA_API CISDI_3DModel Load(const char* path,
                                        ::std::pmr::memory_resource* pMemPool);

}  // namespace IntelliDesign_NS::ModelData