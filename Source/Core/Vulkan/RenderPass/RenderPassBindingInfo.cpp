#include "RenderPassBindingInfo.hpp"

#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.hpp"
#include "Core/Vulkan/Manager/RenderResourceManager.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(Type_PC const& data)
    : value(Type_STLVector<Type_PC> {}) {
    ::std::get<Type_STLVector<Type_PC>>(value).push_back(data);
}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    Type_STLVector<Type_PC> const& data)
    : value(data) {}

RenderPassBindingInfo::RenderPassBindingInfo(Context* context,
                                             RenderResourceManager* resMgr,
                                             PipelineManager* pipelineMgr,
                                             DescriptorSetPool* descPool)
    : pContext(context),
      pResMgr(resMgr),
      pPipelineMgr(pipelineMgr),
      pDescSetPool(descPool),
      mDrawCallMgr {resMgr} {
    mInfos.emplace(sPushConstStr,
                   Type_STLVector<RenderPassBinding::PushContants> {});
}

void RenderPassBindingInfo::SetPipeline(const char* pipelineName,
                                        const char* pipelineLayoutName) {
    mPipelineName = pipelineName;
    if (pipelineLayoutName)
        mPipelineLayoutName = pipelineLayoutName;
    else
        mPipelineLayoutName = pipelineName;

    GeneratePipelineMetaData(mPipelineName);
}

namespace {

auto isPrefix = [](std::string_view prefix, std::string_view full) {
    return prefix == full.substr(0, prefix.size());
};

}  // namespace

RenderPassBindingInfo::Type_BindingValue& RenderPassBindingInfo::operator[](
    const char* name) {
    // Built-in
    if (std::ranges::find_if(
            sBuiltInBindingTypes,
            [name](const char* it) { return strcmp(name, it) == 0; })
        != sBuiltInBindingTypes.end()) {
        return mInfos.at(name);
    }

    // Descriptor set resources
    Type_STLVector<Type_STLString> matched;
    for (auto const& [k, _] : mInfos) {
        if (isPrefix(name, k))
            matched.push_back(k);
    }

    if (!matched.empty()) {
        if (matched.size() == 1) {
            return mInfos.at(matched.front());
        } else {
            // TODO: deal with same name bindings
        }
    }

    throw ::std::runtime_error("not implemented");
}

void RenderPassBindingInfo::OnResize(vk::Extent2D extent) {
    CreateDescriptorSets(nullptr);
    BindDescriptorSets();
    mDrawCallMgr.UpdateArgument_OnResize(extent);
}

void RenderPassBindingInfo::RecordCmd(vk::CommandBuffer cmd) {
    mDrawCallMgr.RecordCmd(cmd);
}

void RenderPassBindingInfo::GenerateMetaData(void* descriptorPNext) {
    for (auto& [name, v] : mInfos) {
        if (name == sPushConstStr) {
            auto& data =
                ::std::get<Type_STLVector<RenderPassBinding::PushContants>>(
                    v.value);
            GeneratePushContantMetaData(data);
        }
    }

    CreateDescriptorSets(descriptorPNext);
    BindDescriptorSets();
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

    auto layout = pPipelineMgr->GetLayout(mPipelineLayoutName.c_str());
    const auto& layoutDatas = layout->GetDescSetLayoutDatas();

    for (auto const& layoutData : layoutDatas) {
        auto data = layoutData->GetData();
        for (uint32_t i = 0; i < data.bindingNames.size(); ++i) {
            mInfos.emplace(data.bindingNames[i], Type_STLString {});
        }
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

void RenderPassBindingInfo::CreateDescriptorSets(void* descriptorPNext) {
    mDescSets.clear();

    auto pipelineLayout = pPipelineMgr->GetLayout(mPipelineLayoutName.c_str());
    const auto& descLayouts = pipelineLayout->GetDescSetLayoutDatas();

    // Create descriptor sets
    for (auto const& descLayout : descLayouts) {
        auto ptr = MakeShared<DescriptorSet>(pContext, descLayout);
        auto requestHandle = pDescSetPool->RequestUnit(descLayout->GetSize());
        ptr->SetRequestedHandle(std::move(requestHandle));
        mDescSets.push_back(ptr);
    }

    // allocate descriptors
    auto allocateDescriptor = [this](DescriptorSet* set, vk::DeviceSize size,
                                     uint32_t binding, vk::DescriptorType type,
                                     vk::DescriptorDataEXT data,
                                     const void* pNext) {
        vk::DescriptorGetInfoEXT descInfo {};
        descInfo.setType(type).setData(data).setPNext(pNext);

        auto resource = set->GetPoolResource();

        pContext->GetDeviceHandle().getDescriptorEXT(
            descInfo, size,
            (char*)resource.hostAddr + resource.offset
                + set->GetBingdingOffset(binding));
    };

    for (uint32_t set = 0; set < descLayouts.size(); ++set) {
        auto descLayout = descLayouts[set];
        auto descSet = mDescSets[set].get();

        for (uint32_t binding = 0; binding < descLayout->GetBindings().size();
             ++binding) {
            auto param = descLayout->GetData().bindingNames[binding];

            auto argument = ::std::get<Type_STLString>(mInfos.at(param).value);
            if (argument.empty())
                continue;

            auto resource = (*pResMgr)[argument.c_str()];
            auto resType = resource->GetType();
            auto descriptorType =
                descLayout->GetBindings()[binding].descriptorType;
            auto descriptorSize = descLayout->GetDescriptorSize(descriptorType);

            if (resType == RenderResource::Type::Buffer) {
                // buffer
                vk::DescriptorAddressInfoEXT bufferInfo {};
                bufferInfo.setAddress(resource->GetBufferDeviceAddress())
                    .setRange(resource->GetBufferSize());
                allocateDescriptor(descSet, descriptorSize, binding,
                                   descriptorType, &bufferInfo,
                                   descriptorPNext);
            } else {
                // texture
                vk::ImageLayout imageLayout;
                bool sampled = false;
                if (descriptorType == vk::DescriptorType::eStorageImage) {
                    imageLayout = vk::ImageLayout::eGeneral;
                } else if (descriptorType
                               == vk::DescriptorType::eCombinedImageSampler
                           || descriptorType
                                  == vk::DescriptorType::eSampledImage) {
                    imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                    sampled = true;
                } else {
                    throw ::std::runtime_error("un-implemented type");
                }

                vk::DescriptorImageInfo imageInfo {};
                imageInfo.setImageView(resource->GetTexViewHandle())
                    .setImageLayout(imageLayout);

                // TODO: sampler setting
                if (sampled) {
                    imageInfo.setSampler(
                        pContext->GetDefaultLinearSamplerHandle());
                }

                allocateDescriptor(descSet, descriptorSize, binding,
                                   descriptorType, &imageInfo, descriptorPNext);
            }
        }
    }
}

void RenderPassBindingInfo::BindDescriptorSets() {
    Type_STLUnorderedSet<vk::DeviceAddress> uniqueBufAddrs;
    uniqueBufAddrs.reserve(mDescSets.size());
    for (auto const& descSet : mDescSets) {
        uniqueBufAddrs.emplace(descSet->GetPoolResource().deviceAddr);
    }

    Type_STLVector<vk::DeviceAddress> bufAddrs {uniqueBufAddrs.begin(),
                                                uniqueBufAddrs.end()};

    Type_STLUnorderedMap<vk::DeviceAddress, uint32_t> bufIdxMap;
    for (uint32_t i = 0; i < bufAddrs.size(); ++i) {
        bufIdxMap.emplace(bufAddrs[i], i);
    }

    mDrawCallMgr.AddArgument_DescriptorBuffer(bufAddrs);

    Type_STLVector<vk::DeviceSize> offsets;
    Type_STLVector<uint32_t> bufIndices;
    for (auto const& descSet : mDescSets) {
        auto resource = descSet->GetPoolResource();
        offsets.push_back(resource.offset);
        bufIndices.push_back(bufIdxMap.at(resource.deviceAddr));
    }

    mDrawCallMgr.AddArgument_DescriptorSet(
        mBindPoint, pPipelineMgr->GetLayoutHandle(mPipelineLayoutName.c_str()),
        0, bufIndices, offsets);
}

}  // namespace IntelliDesign_NS::Vulkan::Core