#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.h"
#include "Core/Vulkan/Native/Descriptors.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Swapchain;
class PipelineManager;
class DescriptorManager;
class RenderResourceManager;

namespace RenderPassBinding {

enum class Type { DSV, RenderInfo, Count };

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
struct TypeTraits<Type::DSV> {
    using value = Type_STLVector<Type_STLString>;
};

template <>
struct TypeTraits<Type::RenderInfo> {
    using value = RenderInfo;
};

template <Type Type>
using TypeTraits_t = typename TypeTraits<Type>::value;

}  // namespace RenderPassBinding

class RenderPassBindingInfo_Barrier;

class RenderPassBindingInfo_Copy;

class RenderPassBindingInfo_Executor;

class RenderPassBindingInfo_PSO {

    class Type_BindingValue {
        using Type_PC = RenderPassBinding::PushContants;
        using Type_RenderInfo = RenderPassBinding::RenderInfo;
        using Type_Value =
            ::std::variant<Type_STLString, Type_PC,
                           Type_STLVector<Type_STLString>, Type_RenderInfo>;

    public:
        Type_BindingValue(const char* str);
        Type_BindingValue(Type_STLString const& str);

        // image name + view name
        Type_BindingValue(::std::initializer_list<Type_STLString> const& str);
        Type_BindingValue(Type_STLVector<Type_STLString> const& str);

        // push constants
        Type_BindingValue(Type_PC const& data);

        // render info
        Type_BindingValue(Type_RenderInfo const& info);

        Type_Value value;
    };

public:
    RenderPassBindingInfo_PSO(Context* context, RenderResourceManager* resMgr,
                          PipelineManager* pipelineMgr,
                          DescriptorSetPool* descPool, Swapchain* sc = nullptr);

    void SetPipeline(const char* pipelineName,
                     const char* pipelineLayoutName = nullptr);

    Type_BindingValue& operator[](RenderPassBinding::Type type);
    Type_BindingValue& operator[](const char* name);
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
    Type_STLUnorderedMap_String<Type_BindingValue> mDescInfos {};
    Type_STLUnorderedMap_String<Type_BindingValue> mPCInfos {};
    Type_STLVector<::std::pair<Type_STLString, Type_BindingValue>> mRTVInfos {};

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
            mBuiltInInfos.emplace(Type,
                                  RenderPassBinding::TypeTraits_t<Type> {});
            return InitBuiltInInfo<static_cast<RenderPassBinding::Type>(
                Utils::EnumCast(Type) + 1)>();
        }
    }
};

}  // namespace IntelliDesign_NS::Vulkan::Core