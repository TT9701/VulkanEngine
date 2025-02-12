/**
 * @file Common.h
 * @author 
 * @brief 该文件定义了 CISDI_3DModel 所需要的基本数据结构
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "BaseTypes.h"

#define CISDI_3DModel_Subfix_Str ".cisdi"
#define CISDI_3DModel_Subfix_WStr L".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

/**
 * @brief CISDI_3DModel 的材质数据结构
 */
struct CISDI_Material {
    INTELLI_DS_PMR_ELEMENT_DEFAULT_CTORS(CISDI_Material, name);

    enum class ShadingModel : uint32_t { Lambert = 0, Phong };

    Type_STLString name;

    struct Data {
        ShadingModel shadingModel {ShadingModel::Lambert};
        float shininess {20.0f};
        Float32_2 padding {};
        Float32_4 ambient {0.2f, 0.2f, 0.2f, 1.0f};
        Float32_4 diffuse {0.6f, 0.6f, 0.6f, 1.0f};
        Float32_4 specular {0.2f, 0.2f, 0.2f, 1.0f};
        Float32_4 emissive {0.0f, 0.0f, 0.0f, 1.0f};
        Float32_4 reflection {0.0f, 0.0f, 0.0f, 1.0f};
        Float32_4 transparency {};
    } data;
};

// vertice attributes
// using SOA for vertex attributes
enum class VertexAttributeEnum : uint8_t { Position, Normal, UV, _Count_ };

/**
 * @brief CISDI_3DModel 的顶点数据结构
 */
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

/**
 * @brief CISDI_3DModel 的 meshlet 数据结构
 */
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

/**
 * @brief CISDI_3DModel 的节点用户自定义属性
 */
using Type_UserPropertyValue =
    ::std::variant<bool, char, unsigned char, int, unsigned int, long long,
                   unsigned long long, float, double, Type_STLString>;

/**
 * @brief CISDI_3DModel 的节点数据结构
 */
struct CISDI_Node {
    INTELLI_DS_PMR_ELEMENT_DEFAULT_CTORS(CISDI_Node, name, childrenIdx,
                                         userProperties);

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