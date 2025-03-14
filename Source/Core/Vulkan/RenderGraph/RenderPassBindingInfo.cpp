#include "RenderPassBindingInfo.h"

#include "Core/Vulkan/Manager/DrawCallManager.h"
#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/RenderResourceManager.h"
#include "Core/Vulkan/Native/DGCSequence.h"
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
    Type_RenderInfo const& info)
    : value(info) {}

RenderPassBindingInfo_PSO::Type_BindingValue::Type_BindingValue(
    Type_BindlessDescInfo const& info)
    : value(info) {}

RenderPassBindingInfo_PSO::RenderPassBindingInfo_PSO(RenderSequence& rs,
                                                     uint32_t index)
    : mRenderSequence(rs), mIndex(index), mDrawCallMgr(rs.mResMgr) {
    InitBuiltInInfos();
}

RenderPassBindingInfo_PSO::RenderPassBindingInfo_PSO(
    RenderSequence& rs, uint32_t index, PipelineLayout const* pipelineLayout)
    : mRenderSequence(rs),
      mIndex(index),
      mDrawCallMgr(rs.mResMgr),
      mPipelineLayout(pipelineLayout) {
    InitBuiltInInfos();
}

void RenderPassBindingInfo_PSO::SetName(const char* name) {
    mName = name;
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

void RenderPassBindingInfo_PSO::GenerateLayoutData() {
    // auto pSeq = mDGCSeqBuf->GetBufferDGCSequence();
    //
    // mBindPoint = pSeq->IsCompute() ? vk::PipelineBindPoint::eCompute
    //                                : vk::PipelineBindPoint::eGraphics;
    //
    // auto pipeLayout = pSeq->GetPipelineLayout();

    mBindPoint = mPipelineLayout->IsCompute()
                   ? vk::PipelineBindPoint::eCompute
                   : vk::PipelineBindPoint::eGraphics;

    const auto layoutDatas = mPipelineLayout->GetDescSetLayoutDatas();

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

    for (auto const& rtv : mPipelineLayout->GetRTVNames()) {
        mRTVInfos.emplace_back(rtv, Type_STLVector<Type_STLString> {});
    }
}

RenderPassBindingInfo_PSO::Type_BindingValue&
RenderPassBindingInfo_PSO::operator[](RenderPassBinding::Type type) {
    return mBuiltInInfos.at(type);
}

namespace {

auto isPrefix = [](Type_STLString_View prefix, Type_STLString_View full) {
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

    // render infos
    Type_STLVector<RenderingAttachmentInfo> colors {};
    RenderingAttachmentInfo depthStencil {};
    RenderPassBinding::RenderInfo renderInfo {};

    // color attachment
    for (auto const& [_, info] : mRTVInfos) {
        const char* imageName;
        const char* viewName;

        auto result =
            ::std::get_if<Type_STLVector<Type_STLString>>(&info.value);
        if (!result) {
            auto result = ::std::get_if<Type_STLString>(&info.value);
            if (!result)
                throw ::std::runtime_error("invalid type");
            else {
                imageName = result->c_str();
                if (Type_STLString {imageName} == "_Swapchain_")
                    break;
                viewName = mRenderSequence.mResMgr[imageName]
                               .GetTexView()
                               ->GetName()
                               .c_str();
            }
        } else {
            if (result->empty())
                break;
            imageName = result->at(0).c_str();
            if (Type_STLString {imageName} == "_Swapchain_")
                break;
            if (result->size() > 1) {
                viewName = result->at(1).c_str();
            } else {
                viewName = mRenderSequence.mResMgr[imageName]
                               .GetTexView()
                               ->GetName()
                               .c_str();
            }
        }

        vk::RenderingAttachmentInfo color {};
        color.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto idx = mRenderSequence.AddRenderResource(imageName);

        color.setImageView(
            mRenderSequence.mResMgr[imageName].GetTexViewHandle(viewName));

        // if (color.loadOp != vk::AttachmentLoadOp::eClear) {
        AddBarrier({idx, vk::ImageLayout::eColorAttachmentOptimal,
                    vk::AccessFlagBits2::eColorAttachmentRead
                        | vk::AccessFlagBits2::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput});
        // }

        colors.emplace_back(imageName, viewName, color);
    }

    for (auto& [type, v] : mBuiltInInfos) {
        switch (type) {
            case RenderPassBinding::Type::DSV: {

                auto result =
                    ::std::get_if<Type_STLVector<Type_STLString>>(&v.value);
                if (!result) {
                    auto result = ::std::get_if<Type_STLString>(&v.value);
                    if (!result)
                        throw ::std::runtime_error("invalid type");
                    else {
                        if (result->empty())
                            break;
                        depthStencil.imageName = *result;
                        depthStencil.viewName =
                            mRenderSequence
                                .mResMgr[depthStencil.imageName.c_str()]
                                .GetTexView()
                                ->GetName();
                    }
                } else {
                    if (result->empty())
                        break;
                    depthStencil.imageName = result->at(0);
                    if (result->size() > 1) {
                        depthStencil.viewName = result->at(1);
                    } else {
                        depthStencil.viewName =
                            mRenderSequence
                                .mResMgr[depthStencil.imageName.c_str()]
                                .GetTexView()
                                ->GetName();
                    }
                }

                depthStencil.info
                    .setImageLayout(
                        vk::ImageLayout::eDepthStencilAttachmentOptimal)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eStore)
                    .setClearValue(vk::ClearDepthStencilValue {0.0f});

                depthStencil.info.setImageView(
                    mRenderSequence.mResMgr[depthStencil.imageName.c_str()]
                        .GetTexViewHandle(depthStencil.viewName.c_str()));

                auto idx = mRenderSequence.AddRenderResource(
                    depthStencil.imageName.c_str());

                // if (depthStencil.info.loadOp != vk::AttachmentLoadOp::eClear) {
                AddBarrier(
                    {idx, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                     vk::AccessFlagBits2::eDepthStencilAttachmentRead
                         | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                     vk::PipelineStageFlagBits2::eLateFragmentTests
                         | vk::PipelineStageFlagBits2::eEarlyFragmentTests});
                // }
                break;
            }
            case RenderPassBinding::Type::RenderInfo: {
                if (mBindPoint == vk::PipelineBindPoint::eCompute)
                    break;

                renderInfo = ::std::get<RenderPassBinding::RenderInfo>(v.value);
                break;
            }
            case RenderPassBinding::Type::DGCSeqBuf: {
                auto& result =
                    ::std::get<Type_STLVector<Type_STLString>>(v.value);

                for (auto const bufName : result) {
                    auto idx =
                        mRenderSequence.AddRenderResource(bufName.c_str());
                    AddBarrier(
                        {.resourceIndex = idx,
                         .layout = vk::ImageLayout::eUndefined,
                         .access = vk::AccessFlagBits2::eIndirectCommandRead,
                         .stages = vk::PipelineStageFlagBits2::eDrawIndirect});

                    mDrawCallMgr.AddArgument_DGCSequence(
                        &mRenderSequence.mResMgr[bufName.c_str()]);
                }
            }

            default: break;
        }
    }

    if (!colors.empty()) {
        mDrawCallMgr.AddArgument_RenderingInfo(
            renderInfo.renderArea, renderInfo.layerCount, renderInfo.viewMask,
            colors, depthStencil);
    }

    // if (mDGCSeqBuf) {
    //     auto idx = mRenderSequence.AddRenderResource(mDGCSeqBuf->GetName());
    //     AddBarrier({idx, vk::ImageLayout::eUndefined,
    //                 vk::AccessFlagBits2::eIndirectCommandRead,
    //                 vk::PipelineStageFlagBits2::eDrawIndirect});
    // }
}

void RenderPassBindingInfo_PSO::Update(const char* resName) {
    // update descriptors
    for (auto const& descInfo : mDescInfos) {
        if (::std::get<Type_STLString>(descInfo.second.value) == resName) {
            auto param = descInfo.first;
            auto const& descLayouts =
                mRenderSequence.mPipelineMgr
                    .GetLayout(mPipelineLayoutName.c_str())
                    ->GetDescSetLayoutDatas();

            ::std::optional<uint32_t> set;
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
            if (!set)
                throw ::std::runtime_error("");

            auto descSize =
                descLayouts[*set]->GetDescriptorSize(binding.descriptorType);

            // TODO: index descriptor that is element array
            return AllocateDescriptor(resName, mDescSets[*set].get(), descSize,
                                      binding.descriptorType, binding.binding,
                                      0, nullptr);
        }
    }

    // update rtv
    Type_STLVector<Type_STLString> rtvCombinedNames;
    for (auto const& rtv : mRTVInfos) {
        auto nameVec =
            ::std::get<Type_STLVector<Type_STLString>>(rtv.second.value);
        if (nameVec[0] == resName) {
            auto combinedName = nameVec[0] + "@" + nameVec[1];
            rtvCombinedNames.push_back(combinedName);
            mDrawCallMgr.UpdateArgument_Attachments(rtvCombinedNames);
            return;
        }
    }
}

void RenderPassBindingInfo_PSO::Update(
    const char* name, RenderPassBinding::BindlessDescBufInfo info) {
    for (auto& [n, _] : mBindlessDescInfos) {
        if (isPrefix(Type_STLString {name}, n)) {
            ::std::get<RenderPassBinding::BindlessDescBufInfo>(
                mBindlessDescInfos.at(n).value) = info;

            Type_STLVector<vk::DeviceAddress> bufAddrs;
            Type_STLVector<vk::DeviceSize> offsets;
            Type_STLVector<uint32_t> bufIndices;
            GenerateDescBufInfos(bufAddrs, offsets, bufIndices);

            for (auto const& descInfo : mBindlessDescInfos) {
                auto bufInfo =
                    ::std::get<RenderPassBinding::BindlessDescBufInfo>(
                        descInfo.second.value);
                bufAddrs.push_back(bufInfo.deviceAddress);
                offsets.push_back(bufInfo.offset);
                bufIndices.push_back(bufAddrs.size() - 1);
            }

            mDrawCallMgr.UpdateArgument_DescriptorBuffer(bufAddrs);

            mDrawCallMgr.UpdateArgument_DescriptorSet(
                mBindPoint,
                mRenderSequence.mPipelineMgr.GetLayoutHandle(
                    mPipelineLayoutName.c_str()),
                0, bufIndices, offsets);
            break;
        }
    }
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
    CMP_NS::Type_STLString_POD name) {
    Type_STLString pipelineName {name};
    if (!pipelineName.empty()) {
        if (mRenderSequence.mPipelineMgr.GetPipelines().contains(
                pipelineName)) {
            auto& pipeline =
                mRenderSequence.mPipelineMgr.GetPipeline(pipelineName.c_str());
            if (pipeline.GetType() == PipelineType::Compute) {
                mBindPoint = vk::PipelineBindPoint::eCompute;
            } else if (pipeline.GetType() == PipelineType::Graphics) {
                mBindPoint = vk::PipelineBindPoint::eGraphics;
            }
            mDrawCallMgr.AddArgument_Pipeline(
                mBindPoint, mRenderSequence.mPipelineMgr.GetPipelineHandle(
                                pipelineName.c_str()));
        } else {
            throw ::std::runtime_error(
                (pipelineName + "Pipeline is not created!").c_str());
        }
    } else {
        throw ::std::runtime_error("pipeline name is empty!");
    }

    auto layout =
        mRenderSequence.mPipelineMgr.GetLayout(mPipelineLayoutName.c_str());
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

    for (auto const& rtv : layout->GetRTVNames()) {
        mRTVInfos.emplace_back(rtv, Type_STLVector<Type_STLString> {});
    }
}

void RenderPassBindingInfo_PSO::GenerateRTVMetaData(
    Type_STLVector<std::array<Type_STLString, 2>> const& data) {}

void RenderPassBindingInfo_PSO::CreateDescriptorSets(void* descriptorPNext) {
    mDescSets.clear();

    Type_STLVector<DescriptorSetLayout*> descLayouts {};
    if (mPipelineLayout) {
        descLayouts = mPipelineLayout->GetDescSetLayoutDatas();
    } else {
        descLayouts =
            mRenderSequence.mPipelineMgr.GetLayout(mPipelineName.c_str())
                ->GetDescSetLayoutDatas();
    }

    // Create descriptor sets
    for (auto const& descLayout : descLayouts) {
        if (descLayout->GetData().bindings[0].descriptorCount > 1)
            continue;
        auto ptr =
            MakeShared<DescriptorSet>(mRenderSequence.mContext, *descLayout);
        auto requestHandle =
            mRenderSequence.mDescPool.RequestUnit(descLayout->GetSize());
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
            auto bindingInfo = descLayout->GetData().bindings[binding];
            auto descCount = bindingInfo.descriptorCount;
            auto descType = bindingInfo.descriptorType;
            auto descSize = descLayout->GetDescriptorSize(descType);

            if (descCount == 1) {
                auto argument =
                    ::std::get<Type_STLString>(mDescInfos.at(param).value);
                if (argument.empty())
                    continue;

                auto idx = mRenderSequence.AddRenderResource(argument.c_str());
                AddBarrier(idx, descType, bindingInfo.stageFlags);

                AllocateDescriptor(argument.c_str(), descSet, descSize,
                                   descType, binding, 0, descriptorPNext);
            } else if (descCount > 1) {
                // bindless descriptors are created before configuring PSO
            }
        }
    }
}

void RenderPassBindingInfo_PSO::BindDescriptorSets() {
    Type_STLVector<vk::DeviceAddress> bufAddrs;
    Type_STLVector<vk::DeviceSize> offsets;
    Type_STLVector<uint32_t> bufIndices;
    GenerateDescBufInfos(bufAddrs, offsets, bufIndices);

    // bindless
    for (auto const& descInfo : mBindlessDescInfos) {
        auto info = ::std::get<RenderPassBinding::BindlessDescBufInfo>(
            descInfo.second.value);
        bufAddrs.push_back(info.deviceAddress);
        offsets.push_back(info.offset);
        bufIndices.push_back(bufAddrs.size() - 1);
    }

    mDrawCallMgr.AddArgument_DescriptorBuffer(bufAddrs);

    vk::PipelineLayout layoutHandle {};
    if (mPipelineLayout) {
        layoutHandle = mPipelineLayout->GetHandle();
    } else {
        layoutHandle = mRenderSequence.mPipelineMgr.GetLayoutHandle(
            mPipelineLayoutName.c_str());
    }

    mDrawCallMgr.AddArgument_DescriptorSet(mBindPoint, layoutHandle, 0,
                                           bufIndices, offsets);
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
        mRenderSequence.mContext.GetDevice()->getDescriptorEXT(
            descInfo, size,
            (char*)resource.hostAddr + resource.offset
                + set->GetBingdingOffset(binding) + idxInBinding * size);
    };

    auto& resource = mRenderSequence.mResMgr[resName];
    auto resType = resource.GetType();

    if (resType == RenderResource::Type::Buffer) {
        // buffer
        vk::DescriptorAddressInfoEXT bufferInfo {};
        bufferInfo.setAddress(resource.GetBufferDeviceAddress())
            .setRange(resource.GetBufferSize());
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
        imageInfo.setImageView(resource.GetTexViewHandle())
            .setImageLayout(imageLayout);

        // TODO: sampler setting
        if (sampled) {
            imageInfo.setSampler(
                mRenderSequence.mContext.GetDefaultLinearSampler().GetHandle());
        }

        getDescriptor(set, descSize, binding, idxInBinding, descriptorType,
                      &imageInfo, pNext);
    }
}

void RenderPassBindingInfo_PSO::GenerateDescBufInfos(
    Type_STLVector<vk::DeviceAddress>& addrs,
    Type_STLVector<vk::DeviceSize>& offsets,
    Type_STLVector<uint32_t>& indices) {
    Type_STLUnorderedSet<vk::DeviceAddress> uniqueBufAddrs;
    uniqueBufAddrs.reserve(mDescSets.size());
    for (auto const& descSet : mDescSets) {
        uniqueBufAddrs.emplace(descSet->GetPoolResource().deviceAddr);
    }

    addrs = Type_STLVector<vk::DeviceAddress> {uniqueBufAddrs.begin(),
                                               uniqueBufAddrs.end()};

    Type_STLUnorderedMap<vk::DeviceAddress, uint32_t> bufIdxMap;
    for (uint32_t i = 0; i < addrs.size(); ++i) {
        bufIdxMap.emplace(addrs[i], i);
    }

    for (auto const& descSet : mDescSets) {
        auto resource = descSet->GetPoolResource();
        offsets.push_back(resource.offset);
        indices.push_back(bufIdxMap.at(resource.deviceAddr));
    }
}

void RenderPassBindingInfo_PSO::AddBarrier(RenderSequence::Barrier const& b) {
    VE_ASSERT(mIndex < mRenderSequence.mPassBarrierInfos.size(), "");

    mRenderSequence.mPassBarrierInfos[mIndex].push_back(b);
}

RenderSequence::Barrier RenderPassBindingInfo_PSO::AddBarrier(
    uint32_t idx, vk::DescriptorType type, vk::ShaderStageFlags shaderStage) {
    RenderSequence::Barrier b {.resourceIndex = idx};

    if (shaderStage & vk::ShaderStageFlagBits::eVertex)
        b.stages |= vk::PipelineStageFlagBits2::eVertexShader;

    if (shaderStage & vk::ShaderStageFlagBits::eTessellationControl)
        b.stages |= vk::PipelineStageFlagBits2::eTessellationControlShader;

    if (shaderStage & vk::ShaderStageFlagBits::eTessellationEvaluation)
        b.stages |= vk::PipelineStageFlagBits2::eTessellationEvaluationShader;

    if (shaderStage & vk::ShaderStageFlagBits::eGeometry)
        b.stages |= vk::PipelineStageFlagBits2::eGeometryShader;

    if (shaderStage & vk::ShaderStageFlagBits::eFragment)
        b.stages |= vk::PipelineStageFlagBits2::eFragmentShader;

    if (shaderStage & vk::ShaderStageFlagBits::eCompute)
        b.stages |= vk::PipelineStageFlagBits2::eComputeShader;

    if (shaderStage & vk::ShaderStageFlagBits::eTaskEXT)
        b.stages |= vk::PipelineStageFlagBits2::eTaskShaderEXT;

    if (shaderStage & vk::ShaderStageFlagBits::eMeshEXT)
        b.stages |= vk::PipelineStageFlagBits2::eMeshShaderEXT;

    switch (type) {
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
            b.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            b.access = vk::AccessFlagBits2::eShaderSampledRead;
            AddBarrier(b);
            break;
        case vk::DescriptorType::eUniformBuffer:
            b.access = vk::AccessFlagBits2::eUniformRead;
            AddBarrier(b);
            break;
        case vk::DescriptorType::eStorageImage:
            b.layout = vk::ImageLayout::eGeneral;
            [[fallthrough]];
        case vk::DescriptorType::eStorageBuffer:
            b.access = vk::AccessFlagBits2::eShaderStorageWrite;
            AddBarrier(b);
            break;
        default: throw ::std::runtime_error("not implemented!");
    }

    return b;
}

RenderPassBindingInfo_Barrier::RenderPassBindingInfo_Barrier(
    VulkanContext& context, RenderResourceManager& resMgr)
    : mContext(context), mResMgr(resMgr), mDrawCallMgr {resMgr} {}

RenderPassBindingInfo_Barrier::RenderPassBindingInfo_Barrier(
    VulkanContext& context, RenderResourceManager& resMgr, Swapchain& sc)
    : mContext(context), mResMgr(resMgr), mDrawCallMgr {resMgr} {}

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
            barrierInfo.setImage(mResMgr[name.c_str()].GetTexHandle());

            bNames.push_back(name);
            imgBarriers.push_back(barrierInfo);
        }
    }
    for (auto const& [name, v] : mBarriers) {
        if (auto barrier = ::std::get_if<BufferBarrier>(&v)) {
            vk::BufferMemoryBarrier2 barrierInfo {};
            barrierInfo.setSrcStageMask(barrier->srcStageMask)
                .setSrcAccessMask(barrier->srcAccessMask)
                .setDstStageMask(barrier->dstStageMask)
                .setDstAccessMask(barrier->dstAccessMask)
                .setSrcQueueFamilyIndex(barrier->srcQueueFamilyIndex)
                .setDstQueueFamilyIndex(barrier->dstQueueFamilyIndex)
                .setBuffer(mResMgr[name.c_str()].GetBufferHandle())
                .setOffset(0)
                .setSize(VK_WHOLE_SIZE);

            bNames.push_back(name);
            bufBarriers.push_back(barrierInfo);
        }
    }
    for (auto const& [name, v] : mBarriers) {
        if (auto barrier = ::std::get_if<MemoryBarrier>(&v)) {
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

void RenderPassBindingInfo_Barrier::OnResize(vk::Extent2D) {
    auto resNames = mResMgr.GetResourceNames_SrcreenSizeRelated();
    Update(resNames);
}

RenderPassBindingInfo_Copy::RenderPassBindingInfo_Copy(RenderSequence& rs,
                                                       uint32_t index)
    : mRenderSequence(rs), mDrawCallMgr(rs.mResMgr), mIndex(index) {}

void RenderPassBindingInfo_Copy::RecordCmd(vk::CommandBuffer cmd) {
    mDrawCallMgr.RecordCmd(cmd);
}

void RenderPassBindingInfo_Copy::GenerateMetaData(void*) {
    auto generateBarrier = [this](CopyInfo const& info) mutable {
        auto srcIdx = mRenderSequence.AddRenderResource(info.src.c_str());
        auto dstIdx = mRenderSequence.AddRenderResource(info.dst.c_str());

        RenderSequence::Barrier srcBarrier {
            srcIdx,
            mRenderSequence.mResMgr[info.src.c_str()].GetType()
                    == RenderResource::Type::Buffer
                ? vk::ImageLayout::eUndefined
                : vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eTransfer};

        RenderSequence::Barrier dstBarrier {
            dstIdx,
            mRenderSequence.mResMgr[info.dst.c_str()].GetType()
                    == RenderResource::Type::Buffer
                ? vk::ImageLayout::eUndefined
                : vk::ImageLayout::eTransferDstOptimal,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eTransfer};

        AddBarrier(srcBarrier);
        AddBarrier(dstBarrier);
    };

    for (auto const& info : mInfos) {
        switch (info.type) {
            case Type_Copy::BufferToBuffer:
                mDrawCallMgr.AddArgument_CopyBufferToBuffer(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferCopy2>(&info.region));
                generateBarrier(info);
                break;
            case Type_Copy::BufferToImage:
                mDrawCallMgr.AddArgument_CopyBufferToImage(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferImageCopy2>(&info.region));
                generateBarrier(info);
                break;
            case Type_Copy::ImageToImage:
                mDrawCallMgr.AddArgument_CopyImageToImage(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::ImageCopy2>(&info.region));
                generateBarrier(info);
                break;
            case Type_Copy::ImageToBuffer:
                mDrawCallMgr.AddArgument_CopyImageToBuffer(
                    info.src.c_str(), info.dst.c_str(),
                    ::std::get_if<vk::BufferImageCopy2>(&info.region));
                generateBarrier(info);
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

void RenderPassBindingInfo_Copy::Update(
    Type_STLVector<Type_STLString> const& resNames) {}

void RenderPassBindingInfo_Copy::OnResize(vk::Extent2D extent) {}

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

void RenderPassBindingInfo_Copy::AddBarrier(RenderSequence::Barrier const& b) {
    VE_ASSERT(mIndex < mRenderSequence.mPassBarrierInfos.size(), "");

    mRenderSequence.mPassBarrierInfos[mIndex].push_back(b);
}

RenderPassBindingInfo_Executor::RenderPassBindingInfo_Executor(
    RenderSequence& rs, uint32_t index)
    : mRenderSequence(rs), mIndex(index) {}

void RenderPassBindingInfo_Executor::RecordCmd(vk::CommandBuffer cmd) {
    for (auto const& infos : mResourceStateInfos) {
        mExecution(cmd, infos);
    }
}

void RenderPassBindingInfo_Executor::GenerateMetaData(void* p) {}

void RenderPassBindingInfo_Executor::Update(const char* resName) {}

void RenderPassBindingInfo_Executor::Update(
    Type_STLVector<Type_STLString> const& resNames) {}

void RenderPassBindingInfo_Executor::OnResize(vk::Extent2D extent) {}

void RenderPassBindingInfo_Executor::AddExecution(
    Type_STLVector<ResourceStateInfos> const& resInfos, Type_Func&& func) {
    for (auto const& resInfo : resInfos) {
        for (auto const& info : resInfo) {
            auto idx = mRenderSequence.AddRenderResource(info.name);
            AddBarrier({idx, info.layout, info.access, info.stages});
        }
    }

    mResourceStateInfos = resInfos;
    mExecution = ::std::move(func);
}

void RenderPassBindingInfo_Executor::AddBarrier(
    RenderSequence::Barrier const& b) {
    VE_ASSERT(mIndex < mRenderSequence.mPassBarrierInfos.size(), "");

    mRenderSequence.mPassBarrierInfos[mIndex].push_back(b);
}

}  // namespace IntelliDesign_NS::Vulkan::Core