#include "DescriptorManager.hpp"

#include "Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

DescriptorManager::DescriptorManager(Context* context, uint32_t bufferCount)
    : pContext(context) {
    vk::PhysicalDeviceProperties2 deviceProp {};
    deviceProp.pNext = &mProperties;
    pContext->GetPhysicalDeviceHandle().getProperties2(&deviceProp);
    CreateDescBuffers(bufferCount);
}

SharedPtr<DescriptorSetLayout> DescriptorManager::CreateDescLayout(
    const char* name,
    Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
    const void* pNext) {

    auto ptr =
        MakeShared<DescriptorSetLayout>(pContext, bindings, mProperties, pNext);

    auto layoutName = ParseDescSetLayoutName(name);
    pContext->SetName(ptr->GetHandle(), layoutName);

    mDescSetLayouts.emplace(layoutName, ptr);

    return ptr;
}

void DescriptorManager::CreateDescLayouts(
    const char* name,
    ::std::initializer_list<DescriptorSetLayoutData> const& datas,
    const void* pNext) {
    Type_STLMap<uint32_t, Type_STLVector<vk::DescriptorSetLayoutBinding>>
        sets {};
    for (auto const& data : datas) {
        sets[data.setIdx].emplace_back(data.bindingIdx, data.type,
                                       data.descCount, data.stage);
    }

    for (auto const& set: sets) {
        Type_STLString setName(name);
        setName.append("_" + ::std::to_string(set.first));
        CreateDescLayout(setName.c_str(), set.second, pNext);
    }
}

vk::DescriptorSetLayout DescriptorManager::GetDescSetLayoutHandle(
    const char* name) const {
    return mDescSetLayouts.at(ParseDescSetLayoutName(name))->GetHandle();
}

DescriptorSetLayout* DescriptorManager::GetDescSetLayout(
    const char* name) const {
    return mDescSetLayouts.at(ParseDescSetLayoutName(name)).get();
}

vk::DeviceAddress DescriptorManager::GetDescBufferAddress(
    uint32_t index) const {
    return mDescBuffers[index].baseAddress;
}

SharedPtr<DescriptorSet> DescriptorManager::CreateDescriptorSet(
    const char* name, const char* setLayoutName, uint32_t bufferIndex) {
    auto setLayout = GetDescSetLayout(setLayoutName);
    auto ptr = MakeShared<DescriptorSet>(pContext, setLayout);
    {
        ::std::unique_lock<::std::mutex> lock {mMtx};
        ptr->SetBufferDatas(bufferIndex,
                            mDescBuffers[bufferIndex].mBufferUsedSize);
        mDescBuffers[bufferIndex].mBufferUsedSize += setLayout->GetSize();
    }
    auto setName = ParseDescSetLayoutName(name);
    mDescSets.emplace(setName, ptr);
    return ptr;
}

DescriptorSet* DescriptorManager::GetDescriptorSet(const char* name) {
    return mDescSets.at(ParseDescSetName(name)).get();
}

void DescriptorManager::CreateBufferDescriptor(
    DescriptorSet* set, uint32_t binding, vk::DescriptorType type,
    vk::DescriptorAddressInfoEXT const* addrInfo, const void* pNext) {
    CreateDescriptor(set, binding, type, addrInfo, pNext);
}

void DescriptorManager::CreateImageDescriptor(
    DescriptorSet* set, uint32_t binding, vk::DescriptorType type,
    vk::DescriptorImageInfo const* imageInfo, const void* pNext) {
    CreateDescriptor(set, binding, type, imageInfo, pNext);
}

void DescriptorManager::BindDescBuffers(vk::CommandBuffer cmd,
                                        ::std::span<uint32_t> bufferIndices) {
    auto bufferCount = bufferIndices.size();
    Type_STLVector<vk::DescriptorBufferBindingInfoEXT> infos(bufferCount);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        infos[i]
            .setUsage(vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT
                      | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT)
            .setAddress(mDescBuffers[bufferIndices[i]].baseAddress);
    }

    cmd.bindDescriptorBuffersEXT(infos);
}

void DescriptorManager::BindDescriptorSets(
    vk::CommandBuffer cmd, vk::PipelineBindPoint bindPoint,
    vk::PipelineLayout layout, uint32_t firstSet,
    ::std::span<uint32_t> bufferIndices,
    ::std::span<Type_STLString> descSetNames) {
    auto descSetCount = descSetNames.size();
    Type_STLVector<vk::DeviceSize> offsets(descSetCount);
    for (uint32_t i = 0; i < descSetCount; ++i) {
        offsets[i] =
            GetDescriptorSet(descSetNames[i].c_str())->GetOffsetInBuffer();
    }

    cmd.setDescriptorBufferOffsetsEXT(bindPoint, layout, firstSet,
                                      bufferIndices, offsets);
}

void DescriptorManager::CreateDescBuffers(uint32_t count) {
    mDescBuffers.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        // TODO: descriptor buffer creation
        mDescBuffers[i].buffer = pContext->CreateStagingBuffer(
            "Descriptors", 1ui64 << 20,
            vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT
                | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT
                | vk::BufferUsageFlagBits::eShaderDeviceAddress
                      & ~vk::BufferUsageFlagBits::eTransferSrc);
        mDescBuffers[i].baseAddress =
            mDescBuffers[i].buffer->GetDeviceAddress();
        mDescBuffers[i].basePtr = mDescBuffers[i].buffer->GetMapPtr();
    }
}

void DescriptorManager::CreateDescriptor(DescriptorSet* set, uint32_t binding,
                                         vk::DescriptorType type,
                                         vk::DescriptorDataEXT const& data,
                                         const void* pNext) {
    vk::DescriptorGetInfoEXT descInfo {};
    descInfo.setType(type).setData(data).setPNext(pNext);

    size_t descSize {0};
    switch (type) {
        case vk::DescriptorType::eSampler:
            descSize = mProperties.samplerDescriptorSize;
            break;
        case vk::DescriptorType::eCombinedImageSampler:
            descSize = mProperties.combinedImageSamplerDescriptorSize;
            break;
        case vk::DescriptorType::eSampledImage:
            descSize = mProperties.sampledImageDescriptorSize;
            break;
        case vk::DescriptorType::eStorageImage:
            descSize = mProperties.storageImageDescriptorSize;
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
            descSize = mProperties.uniformTexelBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageTexelBuffer:
            descSize = mProperties.storageTexelBufferDescriptorSize;
            break;
        case vk::DescriptorType::eUniformBuffer:
            descSize = mProperties.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageBuffer:
            descSize = mProperties.storageBufferDescriptorSize;
            break;
        case vk::DescriptorType::eUniformBufferDynamic:
            descSize = mProperties.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageBufferDynamic:
            descSize = mProperties.storageBufferDescriptorSize;
            break;
        case vk::DescriptorType::eInputAttachment:
            descSize = mProperties.inputAttachmentDescriptorSize;
            break;
        case vk::DescriptorType::eInlineUniformBlock:
            descSize = mProperties.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eAccelerationStructureKHR:
            descSize = mProperties.accelerationStructureDescriptorSize;
            break;
        default:
            throw ::std::runtime_error(
                "invalid ( "
                + VULKAN_HPP_NAMESPACE::toHexString(static_cast<uint32_t>(type))
                + " )");
    }

    pContext->GetDeviceHandle().getDescriptorEXT(
        descInfo, descSize,
        (char*)mDescBuffers[set->GetBufferIndex()].basePtr
            + set->GetOffsetInBuffer() + set->GetBingdingOffset(binding));
}

Type_STLString DescriptorManager::ParseDescSetLayoutName(
    const char* name) const {
    return name;
}

Type_STLString DescriptorManager::ParseDescSetName(const char* name) const {
    return name;
}

}  // namespace IntelliDesign_NS::Vulkan::Core