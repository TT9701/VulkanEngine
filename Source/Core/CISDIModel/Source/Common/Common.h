#pragma once

#include "BaseTypes.h"

#define CISDI_3DModel_Subfix_Str ".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

DeclType_Variant_BasedOnEnum(Type_UserPropertyValue, UserPropertyValueTypeEnum,
                             UserPropertyValueTypeEnum::Count,
                             UserPropertyValueType);

template <class Enum, template <Enum e> class BaseStruct>
class PropertyTuple {
    DeclType_Tuple_BasedOnEnum(Type_PropTuple, Enum, Enum::Count, BaseStruct);

public:
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

private:
    Type_PropTuple datas {};
};

struct CISDI_Vertices {
    using Type_VertexAttribTuple =
        PropertyTuple<VertexAttributeEnum, VertexAttributeType>;

    Type_VertexAttribTuple attributes {};
};

struct CISDI_Material {
    CISDI_Material(::std::pmr::memory_resource* pMemPool) : name {pMemPool} {}

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

struct CISDI_Node {
    CISDI_Node(::std::pmr::memory_resource* pMemPool)
        : name {pMemPool}, childrenIdx {pMemPool}, userProperties {pMemPool} {}

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

#include "MeshletType.h"