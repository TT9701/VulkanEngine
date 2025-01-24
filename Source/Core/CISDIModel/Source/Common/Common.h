#pragma once

#include "BaseTypes.h"

#define CISDI_3DModel_Subfix_Str ".cisdi"
#define CISDI_3DModel_Subfix_WStr L".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

struct CISDI_Material {
    explicit CISDI_Material(::std::pmr::memory_resource* pMemPool)
        : name {pMemPool} {}

    enum class ShadingModel : uint32_t { Lambert = 0, Phong };

    Type_STLString name;

    struct Data {
        ShadingModel shadingModel {ShadingModel::Lambert};
        float shininess {};
        Float32_2 padding {};
        Float32_4 ambient {};
        Float32_4 diffuse {};
        Float32_4 specular {};
        Float32_4 emissive {};
        Float32_4 reflection {};
        Float32_4 transparency {};
    } data;
};

// vertice attributes
// using SOA for vertex attributes
enum class VertexAttributeEnum : uint8_t { Position, Normal, UV, _Count_ };

using CISDI_Vertices =
    PropertyTuple<VertexAttributeEnum, UInt16_3, Int16_2, UInt16_2>;

// meshlet properties
// using SOA for meshlet properties
enum class MeshletPropertyTypeEnum : uint8_t {
    Info,
    Triangle,
    BoundingBox,
    Vertex,
    _Count_
};

using CISDI_Meshlets = PropertyTuple<MeshletPropertyTypeEnum, MeshletInfo,
                                     uint8_t, AABoundingBox, CISDI_Vertices>;

// user properties
enum class UserPropertyValueTypeEnum : uint8_t {
    Bool,
    Char,
    UChar,
    Int,
    UInt,
    LongLong,
    ULongLong,
    Float,
    Double,
    String,
    _Count_
};
using Type_UserPropertyValue =
    ::std::variant<bool, char, unsigned char, int, unsigned int, long long,
                   unsigned long long, float, double, Type_STLString>;

struct CISDI_Node {
    explicit CISDI_Node(::std::pmr::memory_resource* pMemPool)
        : name {pMemPool},
          childrenIdx {::std::pmr::polymorphic_allocator {pMemPool}},
          userProperties {pMemPool} {}

    Type_STLString name;
    uint32_t meshIdx {~0ui32};
    uint32_t materialIdx {~0ui32};
    uint32_t parentIdx {~0ui32};

    uint32_t childCount {0};
    Type_STLVector<uint32_t> childrenIdx;

    uint32_t userPropertyCount {0};
    Type_STLUnorderedMap_String<Type_UserPropertyValue> userProperties;
};

}  // namespace IntelliDesign_NS::ModelData