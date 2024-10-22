#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.hpp"
#include "Core/Vulkan/Native/Descriptors.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Swapchain;
class PipelineManager;
class DescriptorManager;
class RenderResourceManager;

namespace RenderPassBinding {

enum class Type { PushContant, RTV, DSV, RenderInfo, Count };

struct PushContants {
    uint32_t size;
    void* pData;
};

struct RenderInfo {
    vk::Rect2D renderArea;
    uint32_t layerCount;
    uint32_t viewMask;
};

template <Type Type>
struct TypeTraits;

template <>
struct TypeTraits<Type::PushContant> {
    using value = Type_STLVector<PushContants>;
};

template <>
struct TypeTraits<Type::RTV> {
    using value = Type_STLVector<::std::array<Type_STLString, 2>>;
};

template <>
struct TypeTraits<Type::DSV> {
    using value = ::std::array<Type_STLString, 2>;
};

template <>
struct TypeTraits<Type::RenderInfo> {
    using value = RenderInfo;
};

template <Type Type>
using TypeTraits_t = typename TypeTraits<Type>::value;

}  // namespace RenderPassBinding

class RenderPassBindingInfo {

    class Type_BindingValue {
        using Type_PC = RenderPassBinding::PushContants;
        using Type_RenderInfo = RenderPassBinding::RenderInfo;
        using Type_Value =
            ::std::variant<Type_STLVector<Type_PC>,
                           Type_STLVector<::std::array<Type_STLString, 2>>,
                           ::std::array<Type_STLString, 2>, Type_RenderInfo>;

    public:
        // dsv
        Type_BindingValue(::std::array<const char*, 2> const& str);
        Type_BindingValue(::std::array<Type_STLString, 2> const& str);

        // rtvs
        Type_BindingValue(
            ::std::initializer_list<::std::array<const char*, 2>> strs);
        Type_BindingValue(
            Type_STLVector<::std::array<Type_STLString, 2>> const& strs);

        // push constants
        Type_BindingValue(Type_PC const& data);
        Type_BindingValue(Type_STLVector<Type_PC> const& data);

        // render info
        Type_BindingValue(Type_RenderInfo const& info);

        Type_Value value;
    };

public:
    RenderPassBindingInfo(Context* context, RenderResourceManager* resMgr,
                          PipelineManager* pipelineMgr,
                          DescriptorSetPool* descPool, Swapchain* sc = nullptr);

    void SetPipeline(const char* pipelineName,
                     const char* pipelineLayoutName = nullptr);

    Type_BindingValue& operator[](RenderPassBinding::Type type);
    Type_STLString& operator[](const char* name);
    // auto& operator[](EnumType shaderStage);

    void OnResize(vk::Extent2D extent);
    void RecordCmd(vk::CommandBuffer cmd);

    void GenerateMetaData(void* descriptorPNext = nullptr);

    DrawCallManager& GetDrawCallManager();

private:
    void InitBuiltInInfos();

    void GeneratePipelineMetaData(::std::string_view name);
    void GeneratePushContantMetaData(
        Type_STLVector<RenderPassBinding::PushContants> const& data);
    void GenerateRTVMetaData(
        Type_STLVector<::std::array<Type_STLString, 2>> const& data);

    void CreateDescriptorSets(void* descriptorPNext);
    void BindDescriptorSets();

private:
    Context* pContext;
    RenderResourceManager* pResMgr;
    PipelineManager* pPipelineMgr;
    DescriptorSetPool* pDescSetPool;
    Swapchain* pSwapchain;

    DrawCallManager mDrawCallMgr;

    Type_STLUnorderedMap<RenderPassBinding::Type, Type_BindingValue>
        mBuiltInInfos {};
    Type_STLUnorderedMap_String<Type_STLString> mDescInfos {};
    Type_STLString mPipelineName;
    Type_STLString mPipelineLayoutName;
    vk::PipelineBindPoint mBindPoint;

    Type_STLVector<SharedPtr<DescriptorSet>> mDescSets {};

private:
    template <RenderPassBinding::Type Type>
    void InitBuiltInInfo() {
        if constexpr (Type == RenderPassBinding::Type::Count) {
            return;
        } else {
            mBuiltInInfos.emplace(
                Type, typename RenderPassBinding::TypeTraits<Type>::value {});
            return InitBuiltInInfo<static_cast<RenderPassBinding::Type>(
                Utils::EnumCast(Type) + 1)>();
        }
    }
};

}  // namespace IntelliDesign_NS::Vulkan::Core