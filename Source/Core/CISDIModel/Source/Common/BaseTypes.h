#pragma once

#include <cstdint>

#include <memory_resource>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "PMR_Def.h"

namespace IntelliDesign_NS::ModelData {

template <class T>
using Type_STLVector = ::std::pmr::vector<T>;
using Type_STLString = ::std::pmr::string;
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

template <class TEnum, class... TProps>
    requires(sizeof...(TProps) == static_cast<size_t>(TEnum::_Count_))
         && ::std::is_enum_v<TEnum>
class PropertyTuple {
    using Type_PropTuple = ::std::tuple<Type_STLVector<TProps>...>;

public:
    template <TEnum Prop>
    using PropertyType =
        ::std::tuple_element_t<static_cast<size_t>(Prop), Type_PropTuple>;

    template <TEnum Prop>
    auto& GetProperty() {
        return ::std::get<static_cast<size_t>(Prop)>(props);
    }

    template <TEnum Prop>
    const auto& GetProperty() const {
        return ::std::get<static_cast<size_t>(Prop)>(props);
    }

    template <TEnum Prop>
    uint64_t GetProptyByteSize() const {
        return GetProperty<Prop>().size() * sizeof(PropertyType<Prop>);
    }

    template <TEnum Prop>
    const void* GetPropertyPtr() const {
        return reinterpret_cast<const void*>(GetProperty<Prop>().data());
    }

private:
    Type_PropTuple props;
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

// for internal use.
struct InternalMeshData {
    Type_STLVector<Float32_3> positions;
    Type_STLVector<Float32_3> normals;
    Type_STLVector<Float32_2> uvs;
};

// for internal use.
struct InternalMeshlet {
    INTELLI_DS_PMR_ELEMENT_DEFAULT_CTORS(InternalMeshlet, infos, vertIndices,
                                         triangles);

    Type_STLVector<MeshletInfo> infos;
    Type_STLVector<uint32_t> vertIndices;
    Type_STLVector<uint8_t> triangles;
};

}  // namespace IntelliDesign_NS::ModelData