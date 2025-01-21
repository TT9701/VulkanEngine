#pragma once

#include <cstdint>

namespace IntelliDesign_NS::ModelData {
// meshlet properties
// using SOA for meshlet properties
enum class MeshletPropertyTypeEnum : uint8_t {
    Info,
    Triangle,
    BoundingBox,
    Vertex, 
    Count
};

template <MeshletPropertyTypeEnum Type>
struct MeshletPropertyType;

template <>
struct MeshletPropertyType<MeshletPropertyTypeEnum::Info> {
    using Type = Type_STLVector<MeshletInfo>;
};

template <>
struct MeshletPropertyType<MeshletPropertyTypeEnum::Triangle> {
    using Type = Type_STLVector<uint8_t>;
};

template <>
struct MeshletPropertyType<MeshletPropertyTypeEnum::Vertex> {
    using Type = Type_STLVector<Vertices>;
};

template <>
struct MeshletPropertyType<MeshletPropertyTypeEnum::BoundingBox> {
    using Type = Type_STLVector<AABoundingBox>;
};

struct Meshlets {
    using Type_MeshletPropTuple =
        PropertyTuple<MeshletPropertyTypeEnum, MeshletPropertyType>;

    Type_MeshletPropTuple properties {};
};

}  // namespace IntelliDesign_NS::ModelData