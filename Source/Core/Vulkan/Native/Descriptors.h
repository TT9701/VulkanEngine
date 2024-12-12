#pragma once

#include "Core/Utilities/VulkanUtilities.h"

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

#include <Core/System/concurrentqueue.h>

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class Buffer;
class DescriptorSetLayout;

class DescriptorSet {
    using PR_HandleRequest =
        IntelliDesign_NS::Core::AP_MultiConsecutiveUnitManager<
            PoolResource_DescriptorSet>::HandleRequest;

public:
    DescriptorSet(VulkanContext* context, DescriptorSetLayout* setLayout);

    uint32_t GetBindingCount() const;
    vk::DeviceSize GetBingdingOffset(uint32_t binding) const;

    void SetRequestedHandle(PR_HandleRequest&& handle);
    PoolResource GetPoolResource() const;

private:
    DescriptorSetLayout* pSetLayout;

    Type_STLVector<vk::DeviceSize> mBindingOffsets {};

    PR_HandleRequest mResReqHandle;
};

class DescriptorSetLayout {
public:
    struct Data {
        Type_STLVector<Type_STLString> bindingNames;
        Type_STLVector<vk::DescriptorSetLayoutBinding> bindings;
        vk::DeviceSize size {0};
    };

    DescriptorSetLayout(
        VulkanContext* context, Type_STLVector<Type_STLString> const& bindingNames,
        Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
        const void* pNext);
    ~DescriptorSetLayout();

    vk::DescriptorSetLayout GetHandle() const;
    vk::DeviceSize GetSize() const;
    Type_STLVector<vk::DescriptorSetLayoutBinding> const& GetBindings() const;
    Data const& GetData() const;
    size_t GetDescriptorSize(vk::DescriptorType type) const;

private:
    VulkanContext* pContext;

    Data mData;

    vk::DescriptorSetLayout mHandle {};
};

class RenderPassBindingInfo_PSO;

class BindlessDescPool {
public:
    BindlessDescPool(
        VulkanContext* context, Type_STLVector<RenderPassBindingInfo_PSO*> const& pso,
        vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler);

    PoolResource GetPoolResource() const;

    // return texture descriptor idx at bindless set binding.
    uint32_t Add(Texture const* texture);

    // return texture descriptor idx at bindless set binding.
    uint32_t Delete(Texture const* texture);

private:
    void ExpandSet();

private:
    VulkanContext* pContext;
    Type_STLVector<RenderPassBindingInfo_PSO*> mPSOs;

    vk::DescriptorType mDescType;

    SharedPtr<DescriptorSetPool> mDescSetPool;
    SharedPtr<DescriptorSetLayout> mLayout;
    SharedPtr<DescriptorSet> mSet;

    uint32_t mDescCount;
    vk::DeviceSize mDescSize;

    uint32_t mCurrentDescCount {0};
    moodycamel::ConcurrentQueue<uint32_t> mAvailableIndices;
    Type_STLUnorderedMap<Texture const*, uint32_t> mDescIndexMap;
};

}  // namespace IntelliDesign_NS::Vulkan::Core