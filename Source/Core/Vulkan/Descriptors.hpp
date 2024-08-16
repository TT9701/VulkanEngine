#pragma once

#include "Core/Utilities/VulkanUtilities.hpp"

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

struct DescPoolSizeRatio {
    vk::DescriptorType mType;
    float mRatio;
};

namespace __Detail {

class SetLayoutBuilder {
public:
    std::vector<vk::DescriptorSetLayoutBinding> mBindings;

    void AddBinding(uint32_t binding, uint32_t descCount,
                    vk::DescriptorType type);
    void Clear();

    vk::DescriptorSetLayout Build(Context* context,
                                  vk::ShaderStageFlags shaderStages,
                                  vk::DescriptorSetLayoutCreateFlags flags = {},
                                  void* pNext = nullptr);
};

class DescriptorAllocator {
public:
    void InitPool(Context* context, uint32_t initialSets,
                  ::std::span<DescPoolSizeRatio> poolRatios);
    void ClearDescriptors(Context* context);
    void DestroyPool(Context* context);

    vk::DescriptorSet Allocate(Context* context,
                               vk::DescriptorSetLayout layout,
                               void* pNext = nullptr);

private:
    vk::DescriptorPool GetPool(Context* context);

    vk::DescriptorPool CreatePool(Context* context, uint32_t setCount,
                                  std::span<DescPoolSizeRatio> poolRatios);

    ::std::vector<DescPoolSizeRatio> mRatios {};
    ::std::vector<vk::DescriptorPool> mFullPools {};
    ::std::vector<vk::DescriptorPool> mReadyPools {};
    uint32_t mSetsPerPool {};
};

class DescriptorWriter {
public:
    ::std::deque<vk::DescriptorImageInfo> mImageInfos {};
    ::std::deque<vk::DescriptorBufferInfo> mBufferInfos {};
    ::std::vector<vk::WriteDescriptorSet> mWrites {};

    void WriteImage(int binding, vk::DescriptorImageInfo const& imageInfo,
                    vk::DescriptorType type);

    void WriteBuffer(int binding, vk::DescriptorBufferInfo const& bufferInfo,
                     vk::DescriptorType type);

    void Clear();

    void UpdateSet(Context* context, vk::DescriptorSet set);
};

}  // namespace VulkanCore::__Detail

class DescriptorManager {
public:
    DescriptorManager(Context* context, uint32_t initialSets,
                            ::std::span<DescPoolSizeRatio> poolRatio);
    ~DescriptorManager();
    MOVABLE_ONLY(DescriptorManager);

public:
    // Descriptor Set Layout
    void AddDescSetLayoutBinding(uint32_t binding, uint32_t descCount,
                                 vk::DescriptorType type);

    vk::DescriptorSetLayout BuildDescSetLayout(
        ::std::string const& name, vk::ShaderStageFlags shaderStages,
        vk::DescriptorSetLayoutCreateFlags flags = {}, void* pNext = nullptr);

    vk::DescriptorSetLayout GetDescSetLayout(::std::string const& name) const;

    void ClearSetLayout();

    // Descriptors
    vk::DescriptorSet Allocate(::std::string const& name,
                               vk::DescriptorSetLayout layout,
                               void* pNext = nullptr);

    vk::DescriptorSet GetDescriptor(::std::string const& name) const;

    void ClearDescriptors();

    // Descriptor Writes

    void WriteImage(int binding, vk::DescriptorImageInfo imageInfo,
                    vk::DescriptorType type);

    void WriteBuffer(int binding, vk::DescriptorBufferInfo bufferInfo,
                     vk::DescriptorType type);

    void Clear();

    void UpdateSet(vk::DescriptorSet set);

private:
    Context* pContext;

    __Detail::SetLayoutBuilder mSetLayoutBuilder {};

    __Detail::DescriptorAllocator mDescAllocator {};

    __Detail::DescriptorWriter mDescWriter {};

    ::std::unordered_map<::std::string, vk::DescriptorSetLayout> mSetLayouts {};

    // TODO: classify descriptors in different layouts
    ::std::unordered_map<::std::string, vk::DescriptorSet> mDescriptors {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core