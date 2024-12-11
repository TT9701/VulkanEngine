#include "Descriptors.h"

#include "Core/Utilities/Defines.h"
#include "Core/Vulkan/Manager/Context.h"
#include "Core/Vulkan/RenderGraph/RenderPassBindingInfo.h"

namespace IntelliDesign_NS::Vulkan::Core {

DescriptorSet::DescriptorSet(Context* context, DescriptorSetLayout* setLayout)
    : pSetLayout(setLayout) {
    const auto& bindings = setLayout->GetBindings();
    auto bindingCount = bindings.size();

    mBindingOffsets.resize(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i) {
        mBindingOffsets[i] =
            context->GetDevice()->getDescriptorSetLayoutBindingOffsetEXT(
                setLayout->GetHandle(), i);
    }
}

uint32_t DescriptorSet::GetBindingCount() const {
    return mBindingOffsets.size();
}

vk::DeviceSize DescriptorSet::GetBingdingOffset(uint32_t binding) const {
    VE_ASSERT(binding < mBindingOffsets.size(), "Invalid binding");
    return mBindingOffsets[binding];
}

void DescriptorSet::SetRequestedHandle(PR_HandleRequest&& handle) {
    mResReqHandle = std::move(handle);
}

PoolResource DescriptorSet::GetPoolResource() const {
    return mResReqHandle.Get_Resource();
}

DescriptorSetLayout::DescriptorSetLayout(
    Context* context, Type_STLVector<Type_STLString> const& bindingNames,
    Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
    const void* pNext)
    : pContext(context), mData {bindingNames, bindings, 0} {
    vk::DescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo
        .setFlags(vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT)
        .setBindings(bindings)
        .setPNext(pNext);

    mHandle = pContext->GetDevice()->createDescriptorSetLayout(layoutInfo);

    mData.size = Utils::AlignedSize(
        pContext->GetDevice()->getDescriptorSetLayoutSizeEXT(mHandle),
        props.descriptorBufferOffsetAlignment);
}

DescriptorSetLayout::~DescriptorSetLayout() {
    pContext->GetDevice()->destroy(mHandle);
}

vk::DescriptorSetLayout DescriptorSetLayout::GetHandle() const {
    return mHandle;
}

vk::DeviceSize DescriptorSetLayout::GetSize() const {
    return mData.size;
}

Type_STLVector<vk::DescriptorSetLayoutBinding> const&
DescriptorSetLayout::GetBindings() const {
    return mData.bindings;
}

DescriptorSetLayout::Data const& DescriptorSetLayout::GetData() const {
    return mData;
}

size_t DescriptorSetLayout::GetDescriptorSize(vk::DescriptorType type) const {
    auto prop =
        pContext->GetPhysicalDevice()
            .GetProperties<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();

    // pContext->GetPhysicalDevice().GetHandle().getProperties2(&deviceProp);
    size_t descSize {0};
    switch (type) {
        case vk::DescriptorType::eSampler:
            descSize = prop.samplerDescriptorSize;
            break;
        case vk::DescriptorType::eCombinedImageSampler:
            descSize = prop.combinedImageSamplerDescriptorSize;
            break;
        case vk::DescriptorType::eSampledImage:
            descSize = prop.sampledImageDescriptorSize;
            break;
        case vk::DescriptorType::eStorageImage:
            descSize = prop.storageImageDescriptorSize;
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
            descSize = prop.uniformTexelBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageTexelBuffer:
            descSize = prop.storageTexelBufferDescriptorSize;
            break;
        case vk::DescriptorType::eUniformBuffer:
            descSize = prop.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageBuffer:
            descSize = prop.storageBufferDescriptorSize;
            break;
        case vk::DescriptorType::eUniformBufferDynamic:
            descSize = prop.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eStorageBufferDynamic:
            descSize = prop.storageBufferDescriptorSize;
            break;
        case vk::DescriptorType::eInputAttachment:
            descSize = prop.inputAttachmentDescriptorSize;
            break;
        case vk::DescriptorType::eInlineUniformBlock:
            descSize = prop.uniformBufferDescriptorSize;
            break;
        case vk::DescriptorType::eAccelerationStructureKHR:
            descSize = prop.accelerationStructureDescriptorSize;
            break;
        default:
            throw ::std::runtime_error(
                "invalid ( "
                + VULKAN_HPP_NAMESPACE::toHexString(static_cast<uint32_t>(type))
                + " )");
    }
    return descSize;
}

BindlessDescPool::BindlessDescPool(
    Context* context, Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
    vk::DescriptorType type)
    : pContext(context), mPSOs(pso), mDescType(type) {
    auto descBufProps =
        pContext->GetPhysicalDevice()
            .GetProperties<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();

    auto descIndexingProps =
        pContext->GetPhysicalDevice()
            .GetProperties<vk::PhysicalDeviceDescriptorIndexingProperties>();

    mDescCount = ::std::min(
        MAX_BINDLESS_DESCRIPTOR_COUNT,
        descIndexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages);

    mLayout = MakeShared<DescriptorSetLayout>(
        pContext, Type_STLVector<Type_STLString> {"sceneTexs"},
        Type_STLVector<vk::DescriptorSetLayoutBinding> {
            vk::DescriptorSetLayoutBinding {
                0,
                mDescType,
                mDescCount,
                vk::ShaderStageFlagBits::eFragment,
            }},
        descBufProps, nullptr);

    mDescSize = mLayout->GetDescriptorSize(mDescType);
    mDescSetPool = MakeDescSetPoolPtr(pContext, mDescSize * mDescCount);

    mSet = MakeShared<DescriptorSet>(pContext, mLayout.get());
    auto requestHandle = mDescSetPool->RequestUnit(mLayout->GetSize());
    mSet->SetRequestedHandle(std::move(requestHandle));
}

PoolResource BindlessDescPool::GetPoolResource() const {
    return mSet->GetPoolResource();
}

uint32_t BindlessDescPool::Add(Texture const* texture) {
    if (mDescIndexMap.contains(texture))
        return mDescIndexMap.at(texture);

    uint32_t idx;
    bool success = mAvailableIndices.try_dequeue(idx);
    if (!success) {
        if (mCurrentDescCount < mDescCount) {
            idx = mCurrentDescCount++;
        } else {
            ExpandSet();
            return Add(texture);
        }
    }

    auto resource = mSet->GetPoolResource();

    vk::DescriptorImageInfo imageInfo {};
    imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSampler(pContext->GetDefaultNearestSampler().GetHandle());

    imageInfo.setImageView(texture->GetViewHandle());
    vk::DescriptorGetInfoEXT descInfo {};
    descInfo.setType(mDescType).setData(&imageInfo).setPNext(nullptr);

    pContext->GetDevice()->getDescriptorEXT(
        descInfo, mDescSize,
        (char*)resource.hostAddr + resource.offset + mSet->GetBingdingOffset(0)
            + idx * mDescSize);

    mDescIndexMap.emplace(texture, idx);

    return idx;
}

uint32_t BindlessDescPool::Delete(Texture const* texture) {
    uint32_t idx;
    if (mDescIndexMap.contains(texture)) {
        idx = mDescIndexMap.at(texture);

        auto resource = mSet->GetPoolResource();
        auto descSize = mLayout->GetDescriptorSize(mDescType);
        memset((char*)resource.hostAddr + resource.offset
                   + mSet->GetBingdingOffset(0) + idx * descSize,
               0, descSize);

        mAvailableIndices.enqueue(idx);
        mDescIndexMap.erase(texture);
    } else {
        throw ::std::runtime_error(
            "texture is not contained in bindless pool.");
    }
    return idx;
}

void BindlessDescPool::ExpandSet() {
    mDescCount *= 2;
    auto origSize = mLayout->GetSize();

    auto descBufProps =
        pContext->GetPhysicalDevice()
            .GetProperties<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();

    mLayout.reset();
    mLayout = MakeShared<DescriptorSetLayout>(
        pContext, Type_STLVector<Type_STLString> {"sceneTexs"},
        Type_STLVector<vk::DescriptorSetLayoutBinding> {
            vk::DescriptorSetLayoutBinding {
                0,
                mDescType,
                mDescCount,
                vk::ShaderStageFlagBits::eFragment,
            }},
        descBufProps, nullptr);

    auto requestHandle = mDescSetPool->RequestUnit(origSize * 2);
    auto resource = requestHandle.Get_Resource();
    memcpy(resource.hostAddr, mSet->GetPoolResource().hostAddr, origSize);

    mSet.reset();
    mSet = MakeShared<DescriptorSet>(pContext, mLayout.get());
    mSet->SetRequestedHandle(std::move(requestHandle));

    for (auto& pso : mPSOs) {
        pso->Update("sceneTexs", RenderPassBinding::BindlessDescBufInfo {
                                     resource.deviceAddr, resource.offset});
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core