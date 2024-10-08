#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Manager/DrawCallManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PipelineManager;
class DescriptorManager;
class RenderResourceManager;

namespace RenderPassBinding {

struct PushContants {
    uint32_t size;
    void* pData;
};

}  // namespace RenderPassBinding

class RenderPassBindingInfo {
    constexpr static const char* sDescBufStr = "_DescriptorBuffer_";
    constexpr static const char* sPushConstStr = "_PushContants_";

    constexpr static ::std::array sBuiltInBindingTypes = {sDescBufStr,
                                                          sPushConstStr};

    class Type_BindingValue {
        using Type_PC = RenderPassBinding::PushContants;
        using Type_Value =
            ::std::variant<Type_STLString, Type_STLVector<uint32_t>,
                           Type_STLVector<Type_PC>>;

    public:
        Type_BindingValue(const char* str) : value(str) {}

        Type_BindingValue(Type_STLString const& str) : value(str) {}

        Type_BindingValue(::std::initializer_list<uint32_t> const& vec)
            : value(Type_STLVector<uint32_t> {vec}) {}

        Type_BindingValue(Type_STLVector<uint32_t> const& vec) : value(vec) {}

        Type_BindingValue(Type_PC const& data);
        Type_BindingValue(Type_STLVector<Type_PC> const& data);

        Type_Value value;
    };

public:
    RenderPassBindingInfo(RenderResourceManager* resMgr,
                          PipelineManager* pipelineMgr,
                          DescriptorManager* descMgr);

    void Init(const char* pipelineName);

    auto& operator[](const char* name);

    void GenerateMetaData();

    DrawCallManager& GetDrawCallManager();

private:
    void GeneratePipelineMetaData(::std::string_view name);
    void GenerateDescBufMetaData(::std::span<uint32_t> indices);
    void GeneratePushContantMetaData(
        Type_STLVector<RenderPassBinding::PushContants> const& data);

private:
    DrawCallManager mDrawCallMgr;
    PipelineManager* pPipelineMgr;
    DescriptorManager* pDescMgr;

    Type_STLUnorderedMap_String<Type_BindingValue> mInfos {};
    Type_STLString mPipelineName;
    vk::PipelineBindPoint mBindPoint;
};

inline auto& RenderPassBindingInfo::operator[](const char* name) {
    if (std::ranges::find_if(
            sBuiltInBindingTypes,
            [name](const char* it) { return strcmp(name, it) == 0; })
        != sBuiltInBindingTypes.end()) {
        return mInfos.at(name);
    }

    throw ::std::runtime_error("not implemented");
}

}  // namespace IntelliDesign_NS::Vulkan::Core