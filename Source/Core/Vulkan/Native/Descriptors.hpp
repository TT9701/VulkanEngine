#pragma once

#include "Core/Utilities/VulkanUtilities.hpp"

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Buffer;
class DescriptorSetLayout;

class DescriptorSet {
public:
    DescriptorSet(Context* context, DescriptorSetLayout* setLayout);

    uint32_t GetBindingCount() const;
    vk::DeviceSize GetBingdingOffset(uint32_t binding) const;
    vk::DescriptorType GetBingdingType(uint32_t binding) const;
    uint32_t GetBingdingDescCount(uint32_t binding) const;

    void SetBufferDatas(uint32_t idx, vk::DeviceSize offset);
    vk::DeviceSize GetOffsetInBuffer() const;
    uint32_t GetBufferIndex() const;

private:
    DescriptorSetLayout* pSetLayout;

    Type_STLVector<vk::DeviceSize> mBindingOffsets {};

    uint32_t mBufferIndex {0};
    vk::DeviceSize mOffsetInBuffer {0};
};

class DescriptorSetLayout {
public:
    DescriptorSetLayout(
        Context* context,
        Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
        const void* pNext);
    ~DescriptorSetLayout();

    vk::DescriptorSetLayout GetHandle() const;
    vk::DeviceSize GetSize() const;
    Type_STLVector<vk::DescriptorSetLayoutBinding> const& GetBindings() const;

private:
    void CreateDescSetLayout(
        std::span<vk::DescriptorSetLayoutBinding> bindings,
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT const& props,
        const void* pNext);

private:
    Context* pContext;
    Type_STLVector<vk::DescriptorSetLayoutBinding> mBindings;

    vk::DescriptorSetLayout mHandle {};
    vk::DeviceSize mSize {0};
};

}  // namespace IntelliDesign_NS::Vulkan::Core