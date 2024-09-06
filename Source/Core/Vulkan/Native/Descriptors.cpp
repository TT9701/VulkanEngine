#include "Descriptors.hpp"

#include "Core/Vulkan/Manager/Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

DescriptorSet::DescriptorSet(Context* context, DescriptorSetLayout* setLayout)
    : pSetLayout(setLayout) {
    auto bindings = setLayout->GetBindings();
    auto bindingCount = bindings.size();

    mBindingOffsets.resize(bindings.size());
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

vk::DescriptorType DescriptorSet::GetBingdingType(uint32_t binding) const {
    VE_ASSERT(binding < mBindingOffsets.size(), "Invalid binding");
    return pSetLayout->GetBindings()[binding].descriptorType;
}

uint32_t DescriptorSet::GetBingdingDescCount(uint32_t binding) const {
    VE_ASSERT(binding < mBindingOffsets.size(), "Invalid binding");
    return pSetLayout->GetBindings()[binding].descriptorCount;
}

void DescriptorSet::SetBufferDatas(uint32_t idx, vk::DeviceSize offset) {
    mBufferIndex = idx;
    mOffsetInBuffer = offset;
}

vk::DeviceSize DescriptorSet::GetOffsetInBuffer() const {
    return mOffsetInBuffer;
}

uint32_t DescriptorSet::GetBufferIndex() const {
    return mBufferIndex;
}

DescriptorSetLayout::DescriptorSetLayout(
    Context* context,
    Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
    const void* pNext)
    : pContext(context), mBindings(bindings) {
    CreateDescSetLayout(mBindings, props, pNext);
}

DescriptorSetLayout::~DescriptorSetLayout() {
    pContext->GetDeviceHandle().destroy(mHandle);
}

vk::DescriptorSetLayout DescriptorSetLayout::GetHandle() const {
    return mHandle;
}

vk::DeviceSize DescriptorSetLayout::GetSize() const {
    return mSize;
}

Type_STLVector<vk::DescriptorSetLayoutBinding> const&
DescriptorSetLayout::GetBindings() const {
    return mBindings;
}

void DescriptorSetLayout::CreateDescSetLayout(
    std::span<vk::DescriptorSetLayoutBinding> bindings,
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
    const void* pNext) {
    vk::DescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo
        .setFlags(vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT)
        .setBindings(bindings)
        .setPNext(pNext);

    mHandle = pContext->GetDeviceHandle().createDescriptorSetLayout(layoutInfo);

    mSize = Utils::AlignedSize(
        pContext->GetDeviceHandle().getDescriptorSetLayoutSizeEXT(mHandle),
        props.descriptorBufferOffsetAlignment);
}

}  // namespace IntelliDesign_NS::Vulkan::Core