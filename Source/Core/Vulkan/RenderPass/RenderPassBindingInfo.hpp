#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.hpp"
#include "Core/Vulkan/Native/Descriptors.hpp"

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
    constexpr static const char* sPushConstStr = "_PushContants_";




    constexpr static ::std::array sBuiltInBindingTypes = {sPushConstStr};

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
    RenderPassBindingInfo(Context* context, RenderResourceManager* resMgr,
                          PipelineManager* pipelineMgr,
                          DescriptorSetPool* descPool);

    void SetPipeline(const char* pipelineName,
                     const char* pipelineLayoutName = nullptr);

    Type_BindingValue& operator[](const char* name);
    // auto& operator[](EnumType shaderStage);

    void OnResize(vk::Extent2D extent);
    void RecordCmd(vk::CommandBuffer cmd);

    void GenerateMetaData(void* descriptorPNext = nullptr);

    DrawCallManager& GetDrawCallManager();

private:
    void GeneratePipelineMetaData(::std::string_view name);
    void GeneratePushContantMetaData(
        Type_STLVector<RenderPassBinding::PushContants> const& data);

    void CreateDescriptorSets(void* descriptorPNext);
    void BindDescriptorSets();

private:
    Context* pContext;
    RenderResourceManager* pResMgr;
    PipelineManager* pPipelineMgr;
    DescriptorSetPool* pDescSetPool;

    DrawCallManager mDrawCallMgr;

    Type_STLUnorderedMap_String<Type_BindingValue> mInfos {};
    Type_STLString mPipelineName;
    Type_STLString mPipelineLayoutName;
    vk::PipelineBindPoint mBindPoint;

    Type_STLVector<SharedPtr<DescriptorSet>> mDescSets {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core