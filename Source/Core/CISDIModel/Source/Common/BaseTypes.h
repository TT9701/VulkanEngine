#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace IntelliDesign_NS::ModelData {

template <class T>
using Type_STLVector = ::std::pmr::vector<T>;
using Type_STLString = ::std::string;
template <class T>
using Type_STLUnorderedMap_String = std::pmr::unordered_map<Type_STLString, T>;

template <class T, uint32_t Dim>
struct Vec;

template <class T>
struct Vec<T, 2> {
    Vec(T x = (T)0, T y = (T)0) : x(x), y(y) {}

    T x, y;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

template <class T>
struct Vec<T, 3> {
    Vec(T x = (T)0, T y = (T)0, T z = (T)0) : x(x), y(y), z(z) {}

    T x, y, z;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

template <class T>
struct Vec<T, 4> {
    Vec(T x = (T)0, T y = (T)0, T z = (T)0, T w = (T)0)
        : x(x), y(y), z(z), w(w) {}

    T x, y, z, w;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

using Float32_2 = Vec<float, 2>;
using Float32_3 = Vec<float, 3>;
using Float32_4 = Vec<float, 4>;

using UInt16_2 = Vec<uint16_t, 2>;
using UInt16_3 = Vec<uint16_t, 3>;
using Int16_2 = Vec<int16_t, 2>;
using Int16_3 = Vec<int16_t, 3>;

struct Version {
    uint32_t major : 8;
    uint32_t minor : 8;
    uint32_t patch : 16;
};

struct AABoundingBox {
    AABoundingBox()
        : min(FLT_MAX, FLT_MAX, FLT_MAX), max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}

    Float32_3 min;
    Float32_3 max;
};

// exactly same as meshoptimizer::meshopt_Meshlet
struct MeshletInfo {
    uint32_t vertexOffset;
    uint32_t triangleOffset;
    uint32_t vertexCount;
    uint32_t triangleCount;
};

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
    Count
};

template <UserPropertyValueTypeEnum Type>
struct UserPropertyValueType;

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::Bool> {
    using Type = bool;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::Char> {
    using Type = char;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::UChar> {
    using Type = unsigned char;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::Int> {
    using Type = int;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::UInt> {
    using Type = unsigned int;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::LongLong> {
    using Type = long long;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::ULongLong> {
    using Type = unsigned long long;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::Float> {
    using Type = float;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::Double> {
    using Type = double;
};

template <>
struct UserPropertyValueType<UserPropertyValueTypeEnum::String> {
    using Type = ::std::string;
};

// vertice attributes
// using SOA for vertex attributes
enum class VertexAttributeEnum : uint8_t { Position, Normal, UV, Count };

template <VertexAttributeEnum Type>
struct VertexAttributeType;

template <>
struct VertexAttributeType<VertexAttributeEnum::Position> {
    using Type = Type_STLVector<UInt16_3>;
};

template <>
struct VertexAttributeType<VertexAttributeEnum::Normal> {
    using Type = Type_STLVector<Int16_2>;
};

template <>
struct VertexAttributeType<VertexAttributeEnum::UV> {
    using Type = Type_STLVector<UInt16_2>;
};

struct Material {
    enum class ShadingModel : uint32_t { Lambert = 0, Phong };

    Type_STLString name {};

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

// for internal use.
struct InternalMeshData {
    Type_STLVector<Float32_3> positions;
    Type_STLVector<Float32_3> normals;
    Type_STLVector<Float32_2> uvs;
};

// for internal use.
struct InternalMeshlet {
    Type_STLVector<MeshletInfo> infos {};
    Type_STLVector<uint32_t> vertIndices {};
    Type_STLVector<uint8_t> triangles {};
};

#define DeclType_BasedOnEnum(T, UsingType, EnumType, EnumCount, TypeStruct)   \
    using UnderLyingType_##EnumType = ::std::underlying_type_t<EnumType>;     \
    template <size_t... Indices>                           \
    static auto DeclType_##UsingType(                                         \
        ::std::index_sequence<Indices...> const&) {                           \
        return T<                                                             \
            typename TypeStruct<static_cast<EnumType>(Indices)>::Type...> {}; \
    }                                                                         \
                                                                              \
    using UsingType = decltype(DeclType_##UsingType(                          \
        ::std::make_index_sequence<static_cast<UnderLyingType_##EnumType>(    \
            EnumCount)> {}));

#define DeclType_Variant_BasedOnEnum(TypeVariant, EnumType, EnumCount,     \
                                     TypeStruct)                           \
    DeclType_BasedOnEnum(::std::variant, TypeVariant, EnumType, EnumCount, \
                         TypeStruct)

#define DeclType_Tuple_BasedOnEnum(TypeTuple, EnumType, EnumCount, TypeStruct) \
    DeclType_BasedOnEnum(::std::tuple, TypeTuple, EnumType, EnumCount,         \
                         TypeStruct)

}  // namespace IntelliDesign_NS::ModelData