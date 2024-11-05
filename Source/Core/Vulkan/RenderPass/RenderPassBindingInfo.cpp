#include "RenderPassBindingInfo.hpp"

#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.hpp"
#include "Core/Vulkan/Manager/RenderResourceManager.hpp"
#include "Core/Vulkan/Native/Swapchain.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(const char* str)
    : value(Type_STLString {str}) {}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    Type_STLString const& str)
    : value(str) {}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    ::std::initializer_list<Type_STLString> const& str)
    : value(Type_STLVector<Type_STLString> {str}) {}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    Type_STLVector<Type_STLString> const& str)
    : value(str) {}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(Type_PC const& data)
    : value(data) {}

RenderPassBindingInfo::Type_BindingValue::Type_BindingValue(
    Type_RenderInfo const& info)
    : value(info) {}

RenderPassBindingInfo::RenderPassBindingInfo(Context* context,
                                             RenderResourceManager* resMgr,
                                             PipelineManager* pipelineMgr,
                                             DescriptorSetPool* descPool,
                                             Swapchain* sc)
    : pContext(context),
      pResMgr(resMgr),
      pPipelineMgr(pipelineMgr),
      pDescSetPool(descPool),
      pSwapchain(sc),
      mDrawCallMgr {resMgr} {
    InitBuiltInInfos();
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

RenderPassBindingInfo::Type_BindingValue& RenderPassBindingInfo::operator[](
    RenderPassBinding::Type type) {
    return mBuiltInInfos.at(type);
}

namespace {

auto isPrefix = [](std::string_view prefix, std::string_view full) {
    return prefix == full.substr(0, prefix.size());
};

}  // namespace

RenderPassBindingInfo::Type_BindingValue& RenderPassBindingInfo::operator[](
    const char* name) {
    // Descriptor set resources
    Type_STLVector<Type_STLString> matched;
    for (auto const& [k, _] : mDescInfos) {
        if (isPrefix(name, k))
            matched.push_back(k);
    }

    if (!matched.empty()) {
        if (matched.size() == 1) {
            return mDescInfos.at(matched.front());
        } else {
            // TODO: deal with same name bindings
        }
    }

    // push constant resources
    for (auto& pc : mPCInfos) {
        if (pc.first == name)
            return pc.second;
    }

    // rtv resources
    for (auto& rtv : mRTVInfos) {
        if (rtv.first == name) {
            return rtv.second;
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
    // descriptor sets
    CreateDescriptorSets(descriptorPNext);
    BindDescriptorSets();

    // push constant infos
    Type_STLVector<RenderPassBinding::PushContants> pcData;
    for (auto const& pcInfo : mPCInfos) {
        pcData.push_back(
            ::std::get<RenderPassBinding::PushContants>(pcInfo.second.value));
    }
    GeneratePushContantMetaData(pcData);

    // render infos
    Type_STLVector<RenderingAttachmentInfo> colors {};
    RenderingAttachmentInfo depthStencil {};
    RenderPassBinding::RenderInfo renderInfo {};

    // color attachment
    for (auto const& [_, info] : mRTVInfos) {
        auto colorImage =
            ::std::get<Type_STLVector<Type_STLString>>(info.value);

        vk::RenderingAttachmentInfo color {};
        color.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        const char* imageName = colorImage[0].c_str();
        const char* viewName = nullptr;

        if (colorImage[0] == Type_STLString {"_Swapchain_"}) {
            auto idx = pSwapchain->GetCurrentImageIndex();
            viewName = ::std::to_string(idx).c_str();
            color.setImageView(pSwapchain->GetImageViewHandle(idx));
        } else {
            if (!colorImage[1].empty()) {
                viewName = colorImage[1].c_str();
            }
            color.setImageView(
                (*pResMgr)[imageName]->GetTexViewHandle(viewName));
        }

        colors.emplace_back(imageName, viewName, color);
    }

    for (auto& [type, v] : mBuiltInInfos) {
        switch (type) {
            case RenderPassBinding::Type::DSV: {
                auto const& depthImage =
                    ::std::get<Type_STLVector<Type_STLString>>(v.value);
                if (depthImage.empty())
                    break;

                depthStencil.imageName = depthImage[0];

                depthStencil.info
                    .setImageLayout(
                        vk::ImageLayout::eDepthStencilAttachmentOptimal)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eStore)
                    .setClearValue(vk::ClearDepthStencilValue {0.0f});

                if (!depthImage[1].empty()) {
                    depthStencil.viewName = depthImage[1];
                }
                depthStencil.info.setImageView(
                    (*pResMgr)[depthStencil.imageName.c_str()]
                        ->GetTexViewHandle(depthStencil.viewName.c_str()));
                break;
            }
            case RenderPassBinding::Type::RenderInfo: {
                if (mBindPoint == vk::PipelineBindPoint::eCompute)
                    break;

                renderInfo = ::std::get<RenderPassBinding::RenderInfo>(v.value);
                break;
            }
            default: break;
        }
    }

    if (!colors.empty()) {
        mDrawCallMgr.AddArgument_RenderingInfo(
            renderInfo.renderArea, renderInfo.layerCount, renderInfo.viewMask,
            colors, depthStencil);
    }
}

DrawCallManager& RenderPassBindingInfo::GetDrawCallManager() {
    return mDrawCallMgr;
}

void RenderPassBindingInfo::InitBuiltInInfos() {
    InitBuiltInInfo<static_cast<RenderPassBinding::Type>(0)>();
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
            mDescInfos.emplace(data.bindingNames[i], Type_STLString {});
        }
    }

    for (auto const& pc : layout->GetCombinedPushContant()) {
        auto pcName = pc.first;
        mPCInfos.emplace(pcName, RenderPassBinding::PushContants {});
    }

    for (auto const& rtv : layout->GetRTVNames()) {
        mRTVInfos.emplace_back(rtv, Type_STLVector<Type_STLString> {});
    }
}

void RenderPassBindingInfo::GeneratePushContantMetaData(
    Type_STLVector<RenderPassBinding::PushContants> const& data) {
    auto const& ranges =
        pPipelineMgr->GetLayout(mPipelineName.c_str())->GetPCRanges();
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

void RenderPassBindingInfo::GenerateRTVMetaData(
    Type_STLVector<std::array<Type_STLString, 2>> const& data) {}

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

            auto argument =
                ::std::get<Type_STLString>(mDescInfos.at(param).value);
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