#pragma once

#include "Core/Utilities/VulkanUtilities.h"

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/DescriptorSetAllocator.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Buffer;
class DescriptorSetLayout;

class DescriptorSet {
    using PR_HandleRequest =
        IntelliDesign_NS::Core::AP_MultiConsecutiveUnitManager<
            PoolResource_DescriptorSet>::HandleRequest;

public:
    DescriptorSet(Context* context, DescriptorSetLayout* setLayout);

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
        Context* context, Type_STLVector<Type_STLString> const& bindingNames,
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
    Context* pContext;

    Data mData;

    vk::DescriptorSetLayout mHandle {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core