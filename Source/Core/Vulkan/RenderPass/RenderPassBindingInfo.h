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

struct BindlessDescBufInfo {
    vk::DeviceAddress deviceAddress;
    vk::DeviceSize offset;
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

class IRenderPassBindingInfo {
protected:
    IRenderPassBindingInfo() = default;
    virtual ~IRenderPassBindingInfo() = default;

    virtual void RecordCmd(vk::CommandBuffer cmd) = 0;
    virtual void GenerateMetaData(void* descriptorPNext = nullptr) = 0;

    virtual void Update(const char* resName) = 0;
};

class RenderPassBindingInfo_PSO : public IRenderPassBindingInfo {
    class Type_BindingValue {
        using Type_PC = RenderPassBinding::PushContants;
        using Type_RenderInfo = RenderPassBinding::RenderInfo;
        using Type_BindlessDescInfo = RenderPassBinding::BindlessDescBufInfo;
        using Type_Value =
            ::std::variant<Type_STLString, Type_PC,
                           Type_STLVector<Type_STLString>, Type_RenderInfo,
                           Type_BindlessDescInfo>;

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

        // bindless descriptro infos
        Type_BindingValue(Type_BindlessDescInfo const& info);

        Type_Value value;
    };

public:
    RenderPassBindingInfo_PSO(Context* context, RenderResourceManager* resMgr,
                              PipelineManager* pipelineMgr,
                              DescriptorSetPool* descPool,
                              Swapchain* sc = nullptr);
    virtual ~RenderPassBindingInfo_PSO() override = default;

    void SetPipeline(const char* pipelineName,
                     const char* pipelineLayoutName = nullptr);

    Type_BindingValue& operator[](RenderPassBinding::Type type);
    Type_BindingValue& operator[](const char* name);
    // auto& operator[](EnumType shaderStage);

    void OnResize(vk::Extent2D extent);

    virtual void RecordCmd(vk::CommandBuffer cmd) override;
    virtual void GenerateMetaData(void* descriptorPNext = nullptr) override;
    virtual void Update(const char* resName) override;
    void Update(Type_STLVector<Type_STLString> const& names);

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
    void AllocateDescriptor(const char* resName, DescriptorSet* set,
                            size_t descSize, vk::DescriptorType descriptorType,
                            uint32_t binding, uint32_t idxInBinding = 0,
                            void* pNext = nullptr);

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
    Type_STLUnorderedMap_String<Type_BindingValue> mBindlessDescInfos {};
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

class RenderPassBindingInfo_Barrier : public IRenderPassBindingInfo {
public:
    // using whole image subresource as most cases use.
    struct ImageBarrier {
        vk::PipelineStageFlags2 srcStageMask {};
        vk::AccessFlags2 srcAccessMask {};
        vk::PipelineStageFlags2 dstStageMask {};
        vk::AccessFlags2 dstAccessMask {};
        vk::ImageLayout oldLayout {vk::ImageLayout::eUndefined};
        vk::ImageLayout newLayout {vk::ImageLayout::eUndefined};
        uint32_t srcQueueFamilyIndex = {};
        uint32_t dstQueueFamilyIndex = {};
        vk::ImageAspectFlags aspect = {};
    };

    // TODO: buffer barrier
    struct BufferBarrier {};

    // memory barrier is barely used.
    struct MemoryBarrier {};

public:
    RenderPassBindingInfo_Barrier(Context* context,
                                  RenderResourceManager* resMgr,
                                  Swapchain* sc = nullptr);
    virtual ~RenderPassBindingInfo_Barrier() override = default;

    virtual void RecordCmd(vk::CommandBuffer cmd) override;
    virtual void GenerateMetaData(void* p = nullptr) override;
    virtual void Update(const char* resName) override;
    void Update(Type_STLVector<Type_STLString> const& names);

    void AddImageBarrier(const char* name, ImageBarrier const& image);
    void AddBufferBarrier(const char* name, BufferBarrier const& buffer);
    // void AddMemoryBarrier();

private:
    using Type_Barrier =
        ::std::variant<ImageBarrier, BufferBarrier, MemoryBarrier>;

private:
    Context* pContext;
    RenderResourceManager* pResMgr;
    Swapchain* pSwapchain;

    Type_STLVector<::std::pair<Type_STLString, Type_Barrier>> mBarriers;

    DrawCallManager mDrawCallMgr;
};

// only whole resource region copy for now
class RenderPassBindingInfo_Copy : public IRenderPassBindingInfo {
    using Type_CopyRegion =
        DrawCallMetaData<DrawCallMetaDataType::Copy>::Type_CopyRegion;
    using Type_Copy = DrawCallMetaData<DrawCallMetaDataType::Copy>::Type;

    struct CopyInfo {
        Type_STLString src;
        Type_STLString dst;
        Type_Copy type;
        Type_CopyRegion region;
    };

public:
    RenderPassBindingInfo_Copy(RenderResourceManager* resMgr);

    virtual void RecordCmd(vk::CommandBuffer cmd) override;
    virtual void GenerateMetaData(void* p = nullptr) override;
    virtual void Update(const char* resName) override;

    void CopyBufferToBuffer(const char* src, const char* dst,
                            vk::BufferCopy2 const& region);
    void CopyBufferToImage(const char* src, const char* dst,
                           vk::BufferImageCopy2 const& region);
    void CopyImageToImage(const char* src, const char* dst,
                          vk::ImageCopy2 const& region);
    void CopyImageToBuffer(const char* src, const char* dst,
                           vk::BufferImageCopy2 const& region);

private:
    RenderResourceManager* pResMgr;

    DrawCallManager mDrawCallMgr;

    Type_STLVector<CopyInfo> mInfos;
};

class RenderPassBindingInfo_Executor {};

}  // namespace IntelliDesign_NS::Vulkan::Core