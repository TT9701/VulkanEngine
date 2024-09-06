#pragma once

#include <mutex>

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Native/Descriptors.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;
class Buffer;

struct DescriptorSetLayoutData {
    uint32_t setIdx;
    uint32_t bindingIdx;
    vk::DescriptorType type;
    vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eAll;
    uint32_t descCount = 1;
};

class DescriptorManager {
    using Type_DescSetLayouts =
        Type_STLUnorderedMap_String<SharedPtr<DescriptorSetLayout>>;
    using Type_DescSets = Type_STLUnorderedMap_String<SharedPtr<DescriptorSet>>;

public:
    DescriptorManager(Context* context, uint32_t bufferCount = 1);
    ~DescriptorManager() = default;

    SharedPtr<DescriptorSetLayout> CreateDescLayout(
        const char* name,
        Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings,
        const void* pNext = nullptr);

    void CreateDescLayouts(
        const char* name,
        ::std::initializer_list<DescriptorSetLayoutData> const& datas,
        const void* pNext = nullptr);

    vk::DescriptorSetLayout GetDescSetLayoutHandle(const char* name) const;
    DescriptorSetLayout* GetDescSetLayout(const char* name) const;

    vk::DeviceAddress GetDescBufferAddress(uint32_t index = 0) const;

    SharedPtr<DescriptorSet> CreateDescriptorSet(const char* name,
                                                 const char* setLayout,
                                                 uint32_t bufferIndex);

    DescriptorSet* GetDescriptorSet(const char* name);

    void CreateBufferDescriptor(DescriptorSet* set, uint32_t binding,
                                vk::DescriptorType type,
                                vk::DescriptorAddressInfoEXT const* addrInfo,
                                const void* pNext = nullptr);

    void CreateImageDescriptor(DescriptorSet* set, uint32_t binding,
                               vk::DescriptorType type,
                               vk::DescriptorImageInfo const* imageInfo,
                               const void* pNext = nullptr);

    void BindDescBuffers(vk::CommandBuffer cmd,
                         ::std::span<uint32_t> bufferIndices);

    void BindDescriptorSets(vk::CommandBuffer cmd,
                            vk::PipelineBindPoint bindPoint,
                            vk::PipelineLayout layout, uint32_t firstSet,
                            ::std::span<uint32_t> bufferIndices,
                            ::std::span<Type_STLString> descSetNames);

private:
    void CreateDescBuffers(uint32_t count);

    void CreateDescriptor(DescriptorSet* set, uint32_t binding,
                          vk::DescriptorType type,
                          vk::DescriptorDataEXT const& data,
                          const void* pNext = nullptr);

    Type_STLString ParseDescSetLayoutName(const char* name) const;
    Type_STLString ParseDescSetName(const char* name) const;

private:
    Context* pContext;
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT mProperties {};

    Type_DescSetLayouts mDescSetLayouts {};
    Type_DescSets mDescSets {};

    struct DescriptorBuffer {
        SharedPtr<Buffer> buffer;

        vk::DeviceAddress baseAddress;
        void* basePtr = nullptr;
        uint64_t mBufferUsedSize {0};
    };

    Type_STLVector<DescriptorBuffer> mDescBuffers {};

    ::std::mutex mMtx {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core