#pragma once

#include "BaseTypes.h"
#include "Math.h"

#define CISDI_3DModel_Subfix_Str ".cisdi"
#define CISDI_3DModel_Subfix_WStr L".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

DeclType_Variant_BasedOnEnum(Type_UserPropertyValue, UserPropertyValueTypeEnum,
                             UserPropertyValueTypeEnum::Count,
                             UserPropertyValueType);

template <class Enum, template <Enum e> class BaseStruct>
struct PropertyTuple {
    DeclType_Tuple_BasedOnEnum(Type_PropTuple, Enum, Enum::Count, BaseStruct);

    Type_PropTuple datas {};

    template <Enum Prop>
    auto& GetProperty() {
        return ::std::get<static_cast<size_t>(Prop)>(datas);
    }

    template <Enum Prop>
    const auto& GetProperty() const {
        return ::std::get<static_cast<size_t>(Prop)>(datas);
    }

    template <Enum Prop>
    uint64_t GetProptyByteSize() const {
        return GetProperty<Prop>().size()
             * sizeof(::std::tuple_element_t<static_cast<size_t>(Prop),
                                             decltype(datas)>);
    }

    template <Enum Prop>
    const void* GetPropertyPtr() const {
        return reinterpret_cast<const void*>(GetProperty<Prop>().data());
    }
};

struct Vertices {
    using Type_VertexAttribTuple =
        PropertyTuple<VertexAttributeEnum, VertexAttributeType>;

    Type_VertexAttribTuple attributes {};
};

}  // namespace IntelliDesign_NS::ModelData

#include "MeshletType.h"