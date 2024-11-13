#include "Descriptors.h"

#include "Core/Utilities/Defines.h"
#include "Core/Vulkan/Manager/Context.h"

namespace IntelliDesign_NS::Vulkan::Core {

DescriptorSet::DescriptorSet(Context* context, DescriptorSetLayout* setLayout)
    : pSetLayout(setLayout) {
    const auto& bindings = setLayout->GetBindings();
    auto bindingCount = bindings.size();

    mBindingOffsets.resize(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i) {
        mBindingOffsets[i] =
            context->GetDeviceHandle().getDescriptorSetLayoutBindingOffsetEXT(
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

    mHandle = pContext->GetDeviceHandle().createDescriptorSetLayout(layoutInfo);

    mData.size = Utils::AlignedSize(
        pContext->GetDeviceHandle().getDescriptorSetLayoutSizeEXT(mHandle),
        props.descriptorBufferOffsetAlignment);
}

DescriptorSetLayout::~DescriptorSetLayout() {
    pContext->GetDeviceHandle().destroy(mHandle);
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
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT prop;
    vk::PhysicalDeviceProperties2 deviceProp {};
    deviceProp.pNext = &prop;
    pContext->GetPhysicalDeviceHandle().getProperties2(&deviceProp);
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

}  // namespace IntelliDesign_NS::Vulkan::Core