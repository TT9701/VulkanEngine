#pragma once

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

struct PoolResource {
    vk::DeviceAddress deviceAddr {0};
    void* hostAddr {nullptr};
    vk::DeviceSize offset {0};
};

class PoolResource_DescriptorSet {
public:
    PoolResource_DescriptorSet(PoolResource_DescriptorSet const&) = delete;
    PoolResource_DescriptorSet& operator=(PoolResource_DescriptorSet const&) =
        delete;

public:
    PoolResource_DescriptorSet(uint32_t _numBytes_, VulkanContext* context);

    ~PoolResource_DescriptorSet() = default;

public:
    using _Type_Resource_ = PoolResource;

    static _Type_Resource_ _ResourceDefaultValue_();

    void _TransferFromOther_(PoolResource_DescriptorSet& other);

    _Type_Resource_ _Calculate_Resource_(uint32_t idx_Start,
                                         uint32_t idx_End) const;

    size_t _Calculate_ResourceSize_(size_t idx_Start, size_t idx_End) const;

    bool _Calculate_FlagIsValid_() const;

    size_t _Get_numUnits_() const;

private:
    uint32_t mNumBytes {0};

    VulkanContext* pContext;
    SharedPtr<Buffer> pBuffer;
    vk::DeviceAddress baseAddress;
    void* basePtr = nullptr;
};

using DescriptorSetPool =
    IntelliDesign_NS::Core::AP_Pool_NS::AP_ResourcePool_FreeSize<
        PoolResource_DescriptorSet, VulkanContext*>;

/*
 * descriptor set pool min size : 16MB
 */
DescriptorSetPool CreateDescSetPool(VulkanContext& context,
                                    size_t minPoolSize = 1ui64 << 24);

SharedPtr<DescriptorSetPool> MakeDescSetPoolPtr(VulkanContext& context,
                                                size_t minPoolSize = 1ui64
                                                                  << 24);

class DescriptorSetAllocator {
public:
    DescriptorSetAllocator(VulkanContext& context,
                           size_t minPoolSize = 1ui64 << 24);

    PoolResource Allocate(size_t size);

private:
    VulkanContext& pContext;

    SharedPtr<DescriptorSetPool> mPool;
};

}  // namespace IntelliDesign_NS::Vulkan::Core