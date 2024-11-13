#include "RenderPassBindingInfo.h"

#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Native/Swapchain.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(const char* str)
    : value(Type_STLString {str}) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_STLString const& str)
    : value(str) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    ::std::initializer_list<Type_STLString> const& str)
    : value(Type_STLVector<Type_STLString> {str}) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_STLVector<Type_STLString> const& str)
    : value(str) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_PC const& data)
    : value(data) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_RenderInfo const& info)
    : value(info) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_BindlessDescInfo const& info)
    : value(info) {}

RenderPassBindingInfo_PSO::RenderPassBindingInfo_PSO(
    Context* context, RenderResourceManager* resMgr,
    PipelineManager* pipelineMgr, DescriptorSetPool* descPool, Swapchain* sc)
    : pContext(context),
      pResMgr(resMgr),
      pPipelineMgr(pipelineMgr),
      pDescSetPool(descPool),
      pSwapchain(sc),
      mDrawCallMgr {resMgr, sc} {
    InitBuiltInInfos();
}

void RenderPassBindingInfo_PSO::SetPipeline(const char* pipelineName,
                                            const char* pipelineLayoutName) {
    mPipelineName = pipelineName;
    if (pipelineLayoutName)
        mPipelineLayoutName = pipelineLayoutName;
    else
        mPipelineLayoutName = pipelineName;

    GeneratePipelineMetaData(mPipelineName);
}

RenderPassBindingInfo_PSO::Type_BindingValue&
RenderPassBindingInfo_PSO::operator[](RenderPassBinding::Type type) {
    return mBuiltInInfos.at(type);
}

namespace {

auto isPrefix = [](std::string_view prefix, std::string_view full) {
    return prefix == full.substr(0, prefix.size());
};

}  // namespace

RenderPassBindingInfo_PSO::Type_BindingValue&
RenderPassBindingInfo_PSO::operator[](const char* name) {
    // Descriptor set resources
    Type_STLVector<Type_STLString> matched;
    Type_STLString prefix = Type_STLString {name} + "@";
    for (auto const& [k, _] : mDescInfos) {
        if (isPrefix(prefix, k))
            matched.push_back(k);
    }

    if (!matched.empty()) {
        if (matched.size() == 1) {
            return mDescInfos.at(matched.front());
        } else {
            // TODO: deal with same name bindings
        }
    }

    // bindless descriptor buffer info
    for (auto const& [k, _] : mBindlessDescInfos) {
        if (isPrefix(prefix, k))
            matched.push_back(k);
    }

    if (!matched.empty()) {
        if (matched.size() == 1) {
            return mBindlessDescInfos.at(matched.front());
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

void RenderPassBindingInfo_PSO::OnResize(vk::Extent2D extent) {
    CreateDescriptorSets(nullptr);
    BindDescriptorSets();
    mDrawCallMgr.UpdateArgument_OnResize(extent);
}

void RenderPassBindingInfo_PSO::RecordCmd(vk::CommandBuffer cmd) {
    mDrawCallMgr.RecordCmd(cmd);
}

void RenderPassBindingInfo_PSO::GenerateMetaData(void* descriptorPNext) {
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
            viewName = "";
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

void RenderPassBindingInfo_PSO::Update(const char* resName) {
    // update descriptors
    for (auto const& descInfo : mDescInfos) {
        if (::std::get<Type_STLString>(descInfo.second.value) == resName) {
            auto param = descInfo.first;
            auto const& descLayouts =
                pPipelineMgr->GetLayout(mPipelineLayoutName.c_str())
                    ->GetDescSetLayoutDatas();

            uint32_t set = -1;
            vk::DescriptorSetLayoutBinding binding {};
            for (uint32_t s = 0; s < descLayouts.size(); ++s) {
                auto const& data = descLayouts[s]->GetData();
                for (uint32_t b = 0; b < data.size; ++b) {
                    if (data.bindingNames[b] == param) {
                        set = s;
                        binding = data.bindings[b];
                        break;
                    }
                }
            }
            if (set == -1)
                throw ::std::runtime_error("");

            auto descSize =
                descLayouts[set]->GetDescriptorSize(binding.descriptorType);

            // TODO: index descriptor that is element array
            AllocateDescriptor(resName, mDescSets[set].get(), descSize,
                               binding.descriptorType, binding.binding, 0,
                               nullptr);
        }
    }

    // update rtv
    Type_STLVector<Type_STLString> rtvCombinedNames;
    for (auto const& rtv : mRTVInfos) {
        auto nameVec =
            ::std::get<Type_STLVector<Type_STLString>>(rtv.second.value);
        auto combinedName = nameVec[0] + "@" + nameVec[1];
        rtvCombinedNames.push_back(combinedName);
    }
    mDrawCallMgr.UpdateArgument_Attachments(rtvCombinedNames);
}

void RenderPassBindingInfo_PSO::Update(
    Type_STLVector<Type_STLString> const& names) {
    for (auto const& name : names) {
        Update(name.c_str());
    }
}

DrawCallManager& RenderPassBindingInfo_PSO::GetDrawCallManager() {
    return mDrawCallMgr;
}

void RenderPassBindingInfo_PSO::InitBuiltInInfos() {
    InitBuiltInInfo<static_cast<RenderPassBinding::Type>(0)>();
}

void RenderPassBindingInfo_PSO::GeneratePipelineMetaData(
    ::std::string_view name) {
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
        for (uint32_t i = 0; i < data.bindings.size(); ++i) {
            auto descCount = data.bindings[i].descriptorCount;
            if (descCount == 1) {
                mDescInfos.emplace(data.bindingNames[i], Type_STLString {});
            } else if (descCount > 1) {
                // for (uint32_t j = 0; j < descCount; ++j) {
                //     auto bindingName = data.bindingNames[i];
                //     auto idx = bindingName.find("@");
                //     bindingName.insert(idx, ::std::to_string(j));
                //     mDescInfos.emplace(bindingName, Type_STLString {});
                // }
                mBindlessDescInfos.emplace(
                    data.bindingNames[i],
                    RenderPassBinding::BindlessDescBufInfo {});
            }
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

void RenderPassBindingInfo_PSO::GeneratePushContantMetaData(
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

void RenderPassBindingInfo_PSO::GenerateRTVMetaData(
    Type_STLVector<std::array<Type_STLString, 2>> const& data) {}

void RenderPassBindingInfo_PSO::CreateDescriptorSets(void* descriptorPNext) {
    mDescSets.clear();

    auto pipelineLayout = pPipelineMgr->GetLayout(mPipelineLayoutName.c_str());
    const auto& descLayouts = pipelineLayout->GetDescSetLayoutDatas();

    // Create descriptor sets
    for (auto const& descLayout : descLayouts) {
        if (descLayout->GetData().bindings[0].descriptorCount > 1)
            continue;
        auto ptr = MakeShared<DescriptorSet>(pContext, descLayout);
        auto requestHandle = pDescSetPool->RequestUnit(descLayout->GetSize());
        ptr->SetRequestedHandle(std::move(requestHandle));
        mDescSets.push_back(ptr);
    }

    for (uint32_t set = 0; set < descLayouts.size(); ++set) {
        auto descLayout = descLayouts[set];
        if (descLayout->GetData().bindings[0].descriptorCount > 1)
            continue;
        auto descSet = mDescSets[set].get();

        for (uint32_t binding = 0; binding < descLayout->GetBindings().size();
             ++binding) {
            auto param = descLayout->GetData().bindingNames[binding];
            auto descCount =
                descLayout->GetData().bindings[binding].descriptorCount;
            auto descType =
                descLayout->GetData().bindings[binding].descriptorType;
            auto descSize = descLayout->GetDescriptorSize(descType);

            if (descCount == 1) {
                auto argument =
                    ::std::get<Type_STLString>(mDescInfos.at(param).value);
                if (argument.empty())
                    continue;

                AllocateDescriptor(argument.c_str(), descSet, descSize,
                                   descType, binding, 0, descriptorPNext);
            } else if (descCount > 1) {
                // bindless descriptors are created before configuring PSO
            }
        }
    }
}

void RenderPassBindingInfo_PSO::BindDescriptorSets() {
    Type_STLUnorderedSet<vk::DeviceAddress> uniqueBufAddrs;
    uniqueBufAddrs.reserve(mDescSets.size());
    for (auto const& descSet : mDescSets) {
        uniqueBufAddrs.emplace(descSet->GetPoolResource().deviceAddr);
    }

    // bindless
    for (auto const& descInfo : mBindlessDescInfos) {
        auto info = ::std::get<RenderPassBinding::BindlessDescBufInfo>(
            descInfo.second.value);
        uniqueBufAddrs.emplace(info.deviceAddress);
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

    // bindless
    for (auto const& descInfo : mBindlessDescInfos) {
        auto info = ::std::get<RenderPassBinding::BindlessDescBufInfo>(
            descInfo.second.value);
        offsets.push_back(info.offset);
        bufIndices.push_back(bufIdxMap.at(info.deviceAddress));
    }

    mDrawCallMgr.AddArgument_DescriptorSet(
        mBindPoint, pPipelineMgr->GetLayoutHandle(mPipelineLayoutName.c_str()),
        0, bufIndices, offsets);
}

void RenderPassBindingInfo_PSO::AllocateDescriptor(
    const char* resName, DescriptorSet* set, size_t descSize,
    vk::DescriptorType descriptorType, uint32_t binding, uint32_t idxInBinding,
    void* pNext) {
    auto getDescriptor = [this](DescriptorSet* set, vk::DeviceSize size,
                                uint32_t binding, uint32_t idxInBinding,
                                vk::DescriptorType type,
                                vk::DescriptorDataEXT data, const void* pNext) {
        vk::DescriptorGetInfoEXT descInfo {};
        descInfo.setType(type).setData(data).setPNext(pNext);

        auto resource = set->GetPoolResource();

        // location = bufferAddr + setOffset + descriptorOffset + (arrayElement * descSize)
        pContext->GetDeviceHandle().getDescriptorEXT(
            descInfo, size,
            (char*)resource.hostAddr + resource.offset
                + set->GetBingdingOffset(binding) + idxInBinding * size);
    };

    auto resource = (*pResMgr)[resName];
    auto resType = resource->GetType();

    if (resType == RenderResource::Type::Buffer) {
        // buffer
        vk::DescriptorAddressInfoEXT bufferInfo {};
        bufferInfo.setAddress(resource->GetBufferDeviceAddress())
            .setRange(resource->GetBufferSize());
        getDescriptor(set, descSize, binding, idxInBinding, descriptorType,
                      &bufferInfo, pNext);
    } else {
        // texture
        vk::ImageLayout imageLayout;
        bool sampled = false;
        if (descriptorType == vk::DescriptorType::eStorageImage) {
            imageLayout = vk::ImageLayout::eGeneral;
        } else if (descriptorType == vk::DescriptorType::eCombinedImageSampler
                   || descriptorType == vk::DescriptorType::eSampledImage) {
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
            imageInfo.setSampler(pContext->GetDefaultLinearSamplerHandle());
        }

        getDescriptor(set, descSize, binding, idxInBinding, descriptorType,
                      &imageInfo, pNext);
    }
}

RenderPassBindingInfo_Barrier::RenderPassBindingInfo_Barrier(
    Context* context, RenderResourceManager* resMgr, Swapchain* sc)
    : pContext(context),
      pResMgr(resMgr),
      pSwapchain(sc),
      mDrawCallMgr {resMgr, sc} {}

void RenderPassBindingInfo_Barrier::RecordCmd(vk::CommandBuffer cmd) {
    mDrawCallMgr.RecordCmd(cmd);
}

void RenderPassBindingInfo_Barrier::GenerateMetaData(void*) {
    Type_STLVector<Type_STLString> bNames;
    Type_STLVector<vk::ImageMemoryBarrier2> imgBarriers;
    Type_STLVector<vk::BufferMemoryBarrier2> bufBarriers;
    Type_STLVector<vk::MemoryBarrier2> memBarriers;

    for (auto const& [name, v] : mBarriers) {
        if (auto barrier = ::std::get_if<ImageBarrier>(&v)) {
            vk::ImageMemoryBarrier2 barrierInfo {};
            barrierInfo.setSrcStageMask(barrier->srcStageMask)
                .setSrcAccessMask(barrier->srcAccessMask)
                .setDstStageMask(barrier->dstStageMask)
                .setDstAccessMask(barrier->dstAccessMask)
                .setOldLayout(barrier->oldLayout)
                .setNewLayout(barrier->newLayout)
                .setSrcQueueFamilyIndex(barrier->srcQueueFamilyIndex)
                .setDstQueueFamilyIndex(barrier->dstQueueFamilyIndex)
                .setSubresourceRange(
                    Utils::GetWholeImageSubresource(barrier->aspect));
            if (name == "_Swapchain_") {
                barrierInfo.setImage(
                    pSwapchain->GetCurrentImage().GetTexHandle());
            } else {
                barrierInfo.setImage((*pResMgr)[name.c_str()]->GetTexHandle());
            }

            bNames.push_back(name);
            imgBarriers.push_back(barrierInfo);

        } else if (auto barrier = ::std::get_if<BufferBarrier>(&v)) {
            vk::BufferMemoryBarrier2 barrierInfo {};

            // TODO: buffer barrier

            bNames.push_back(name);
            bufBarriers.push_back(barrierInfo);
        } else if (auto barrier = ::std::get_if<MemoryBarrier>(&v)) {
            vk::MemoryBarrier2 barrierInfo {};

            // TODO: memory barrier

            bNames.push_back(name);
            memBarriers.push_back(barrierInfo);
        }
    }

    mDrawCallMgr.AddArgument_Barriers(bNames, imgBarriers, memBarriers,
                                      bufBarriers);
}

void RenderPassBindingInfo_Barrier::AddImageBarrier(const char* name,
                                                    ImageBarrier const& image) {
    mBarriers.emplace_back(name, image);
}

void RenderPassBindingInfo_Barrier::AddBufferBarrier(
    const char* name, BufferBarrier const& buffer) {
    mBarriers.emplace_back(name, buffer);
}

void RenderPassBindingInfo_Barrier::Update(const char* name) {
    auto it = ::std::find_if(
        mBarriers.cbegin(), mBarriers.cend(),
        [name](::std::pair<Type_STLString, Type_Barrier> const& t) {
            return t.first == name;
        });

    if (it != mBarriers.cend()) {
        mDrawCallMgr.UpdateArgument_Barriers({name});
    }
}

void RenderPassBindingInfo_Barrier::Update(
    Type_STLVector<Type_STLString> const& names) {
    for (auto const& name : names) {
        Update(name.c_str());
    }
}

RenderPassBindingInfo_Copy::RenderPassBindingInfo_Copy(
    RenderResourceManager* resMgr)
    : pResMgr(resMgr), mDrawCallMgr(resMgr) {}

void RenderPassBindingInfo_Copy::RecordCmd(vk::CommandBuffer cmd) {
    mDrawCallMgr.RecordCmd(cmd);
}

void RenderPassBindingInfo_Copy::GenerateMetaData(void*) {
    for (auto const& info : mInfos) {
        switch (info.type) {
            case Type_Copy::BufferToBuffer:
                mDrawCallMgr.AddArgument_CopyBufferToBuffer(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferCopy2>(&info.region));
                break;
            case Type_Copy::BufferToImage:
                mDrawCallMgr.AddArgument_CopyBufferToImage(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferImageCopy2>(&info.region));
                break;
            case Type_Copy::ImageToImage:
                mDrawCallMgr.AddArgument_CopyImageToImage(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::ImageCopy2>(&info.region));
                break;
            case Type_Copy::ImageToBuffer:
                mDrawCallMgr.AddArgument_CopyImageToBuffer(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferImageCopy2>(&info.region));
                break;
        }
    }
}

void RenderPassBindingInfo_Copy::Update(const char* resName) {
    for (uint32_t i = 0; i < mInfos.size(); ++i) {
        if (resName == mInfos[i].src) {
            mDrawCallMgr.UpdateArgument_CopySrc(resName, i);
        } else if (resName == mInfos[i].dst) {
            mDrawCallMgr.UpdateArgument_CopyDst(resName, i);
        }
    }
}

void RenderPassBindingInfo_Copy::CopyBufferToBuffer(
    const char* src, const char* dst, vk::BufferCopy2 const& region) {
    mInfos.emplace_back(src, dst, Type_Copy::BufferToBuffer, region);
}

void RenderPassBindingInfo_Copy::CopyBufferToImage(
    const char* src, const char* dst, vk::BufferImageCopy2 const& region) {
    mInfos.emplace_back(src, dst, Type_Copy::BufferToImage, region);
}

void RenderPassBindingInfo_Copy::CopyImageToImage(
    const char* src, const char* dst, vk::ImageCopy2 const& region) {
    mInfos.emplace_back(src, dst, Type_Copy::ImageToImage, region);
}

void RenderPassBindingInfo_Copy::CopyImageToBuffer(
    const char* src, const char* dst, vk::BufferImageCopy2 const& region) {
    mInfos.emplace_back(src, dst, Type_Copy::BufferToBuffer, region);
}

}  // namespace IntelliDesign_NS::Vulkan::Core