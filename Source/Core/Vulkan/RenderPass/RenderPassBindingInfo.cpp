#include "RenderPassBindingInfo.hpp"

#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(Type_PC const& data)
    : value(Type_STLVector<Type_PC> {}) {
    ::std::get<Type_STLVector<Type_PC>>(value).push_back(data);
}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    Type_STLVector<Type_PC> const& data)
    : value(data) {}

RenderPassBindingInfo::RenderPassBindingInfo(RenderResourceManager* resMgr,
                                             PipelineManager* pipelineMgr,
                                             DescriptorManager* descMgr)
    : mDrawCallMgr {resMgr}, pPipelineMgr(pipelineMgr), pDescMgr(descMgr) {
    mInfos.emplace(sDescBufStr, Type_STLVector<uint32_t> {});
    mInfos.emplace(sPushConstStr,
                   Type_STLVector<RenderPassBinding::PushContants> {});
}

void RenderPassBindingInfo::Init(const char* pipelineName) {
    mPipelineName = pipelineName;
    GeneratePipelineMetaData(mPipelineName);
}

void RenderPassBindingInfo::GenerateMetaData() {
    for (auto& [name, v] : mInfos) {
        if (name == sDescBufStr) {
            auto& indices = ::std::get<Type_STLVector<uint32_t>>(v.value);
            GenerateDescBufMetaData(indices);
            continue;
        }
        if (name == sPushConstStr) {
            auto& data =
                ::std::get<Type_STLVector<RenderPassBinding::PushContants>>(
                    v.value);
            GeneratePushContantMetaData(data);
        }
    }
}

DrawCallManager& RenderPassBindingInfo::GetDrawCallManager() {
    return mDrawCallMgr;
}

void RenderPassBindingInfo::GeneratePipelineMetaData(::std::string_view name) {
    Type_STLString pipelineName {name};
    if (!pipelineName.empty()) {
        if (pPipelineMgr->GetComputePipelines().contains(pipelineName)) {
            mBindPoint = vk::PipelineBindPoint::eCompute;
            mDrawCallMgr.AddArgument_Pipeline(
                mBindPoint,
                pPipelineMgr->GetComputePipelineHandle(pipelineName.c_str()));
        } else if (pPipelineMgr->GetGraphicsPipelines().contains(
                       pipelineName)) {
            mBindPoint = vk::PipelineBindPoint::eGraphics;
            mDrawCallMgr.AddArgument_Pipeline(
                mBindPoint,
                pPipelineMgr->GetGraphicsPipelineHandle(pipelineName.c_str()));
        }
    } else {
        throw ::std::runtime_error("pipeline name is empty!");
    }
}

void RenderPassBindingInfo::GenerateDescBufMetaData(
    std::span<uint32_t> indices) {
    if (!indices.empty()) {
        Type_STLVector<vk::DeviceAddress> addresses;
        addresses.reserve(indices.size());
        for (auto const& index : indices) {
            addresses.push_back(pDescMgr->GetDescBufferAddress(index));
        }
        mDrawCallMgr.AddArgument_DescriptorBuffer(addresses);
    }
}

void RenderPassBindingInfo::GeneratePushContantMetaData(
    Type_STLVector<RenderPassBinding::PushContants> const& data) {
    auto const& ranges =
        pPipelineMgr->GetLayout(mPipelineName.c_str())->GetPushConstants();
    uint32_t count = ranges.size();
    assert(count == data.size());

    auto layout = pPipelineMgr->GetLayoutHandle(mPipelineName.c_str());

    for (uint32_t i = 0; i < count; ++i) {
        auto layoutRangeSize = ranges[i].size;
        auto dataRangeSize = data[i].size;
        assert(layoutRangeSize == dataRangeSize);

        mDrawCallMgr.AddArgument_PushConstant(layout, ranges[i].stageFlags,
                                              ranges[i].offset, layoutRangeSize,
                                              data[i].pData);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core