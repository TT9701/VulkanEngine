#include "DescriptorSetAllocator.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

PoolResource_DescriptorSet::PoolResource_DescriptorSet(uint32_t _numBytes_,
                                                       Context* context)
    : mNumBytes(_numBytes_), pContext(context) {
    vk::BufferUsageFlags usage {
        vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT
        | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT
        | vk::BufferUsageFlagBits::eShaderDeviceAddress
              & ~vk::BufferUsageFlagBits::eTransferSrc};

    pBuffer =
        pContext->CreateStagingBuffer("DescriptorSets", _numBytes_, usage);

    baseAddress = pBuffer->GetDeviceAddress();
    basePtr = pBuffer->GetMapPtr();
}

PoolResource_DescriptorSet::_Type_Resource_
PoolResource_DescriptorSet::_ResourceDefaultValue_() {
    return _Type_Resource_ {};
}

void PoolResource_DescriptorSet::_TransferFromOther_(
    PoolResource_DescriptorSet& other) {}

PoolResource_DescriptorSet::_Type_Resource_
PoolResource_DescriptorSet::_Calculate_Resource_(uint32_t idx_Start,
                                                 uint32_t idx_End) const {
    return _Type_Resource_ {baseAddress, basePtr, idx_Start};
}

size_t PoolResource_DescriptorSet::_Calculate_ResourceSize_(
    size_t idx_Start, size_t idx_End) const {
    return idx_End - idx_Start + 1;
}

bool PoolResource_DescriptorSet::_Calculate_FlagIsValid_() const {
    return baseAddress != 0;
}

size_t PoolResource_DescriptorSet::_Get_numUnits_() const {
    return mNumBytes;
}

DescriptorSetPool CreateDescriptorSetPool(Context* context,
                                          size_t minPoolSize) {
#ifndef NDEBUG
    return DescriptorSetPool {true, true, minPoolSize, context};
#else
    return DescriptorSetPool {false, true, minPoolSize, context};
#endif
}

DescriptorSetAllocator::DescriptorSetAllocator(Context* context,
                                               DescriptorSetPool* pool)
    : pContext(context), mPool(pool) {}

PoolResource DescriptorSetAllocator::Allocate(size_t size) {
    auto handle = mPool->RequestUnit(size);
    return handle.Get_Resource();
}

}  // namespace IntelliDesign_NS::Vulkan::Core