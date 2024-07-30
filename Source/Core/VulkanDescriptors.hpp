#pragma once

#include "Core/Utilities/VulkanUtilities.hpp"

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanContext;

struct DescPoolSizeRatio {
    vk::DescriptorType mType;
    float              mRatio;
};

namespace VulkanCore::__Detail {

class SetLayoutBuilder {
public:
    std::vector<vk::DescriptorSetLayoutBinding> mBindings;

    void AddBinding(uint32_t binding, uint32_t descCount,
                    vk::DescriptorType type);
    void Clear();

    vk::DescriptorSetLayout Build(VulkanContext*       context,
                                  vk::ShaderStageFlags shaderStages,
                                  vk::DescriptorSetLayoutCreateFlags flags = {},
                                  void* pNext = nullptr);
};

class DescriptorAllocator {
public:
    void InitPool(VulkanContext* context, uint32_t initialSets,
                  ::std::span<DescPoolSizeRatio> poolRatios);
    void ClearDescriptors(VulkanContext* context);
    void DestroyPool(VulkanContext* context);

    vk::DescriptorSet Allocate(VulkanContext*          context,
                               vk::DescriptorSetLayout layout,
                               void*                   pNext = nullptr);

private:
    vk::DescriptorPool GetPool(VulkanContext* context);

    vk::DescriptorPool CreatePool(VulkanContext* context, uint32_t setCount,
                                  std::span<DescPoolSizeRatio> poolRatios);

    ::std::vector<DescPoolSizeRatio>  mRatios {};
    ::std::vector<vk::DescriptorPool> mFullPools {};
    ::std::vector<vk::DescriptorPool> mReadyPools {};
    uint32_t                          mSetsPerPool {};
};

class DescriptorWriter {
public:
    ::std::deque<vk::DescriptorImageInfo>  mImageInfos {};
    ::std::deque<vk::DescriptorBufferInfo> mBufferInfos {};
    ::std::vector<vk::WriteDescriptorSet>  mWrites {};

    void WriteImage(int binding, vk::DescriptorImageInfo const& imageInfo,
                    vk::DescriptorType type);

    void WriteBuffer(int binding, vk::DescriptorBufferInfo const& bufferInfo,
                     vk::DescriptorType type);

    void Clear();

    void UpdateSet(VulkanContext* context, vk::DescriptorSet set);
};

}  // namespace VulkanCore::__Detail

class VulkanDescriptorManager {
public:
    VulkanDescriptorManager(VulkanContext* context, uint32_t initialSets,
                            ::std::span<DescPoolSizeRatio> poolRatio);
    ~VulkanDescriptorManager();
    MOVABLE_ONLY(VulkanDescriptorManager);

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
    vk::DescriptorSet Allocate(::std::string const&    name,
                               vk::DescriptorSetLayout layout,
                               void*                   pNext = nullptr);

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
    VulkanContext* pContext;

    VulkanCore::__Detail::SetLayoutBuilder mSetLayoutBuilder {};

    VulkanCore::__Detail::DescriptorAllocator mDescAllocator {};

    VulkanCore::__Detail::DescriptorWriter mDescWriter {};

    ::std::unordered_map<::std::string, vk::DescriptorSetLayout> mSetLayouts {};

    // TODO: classify descriptors in different layouts
    ::std::unordered_map<::std::string, vk::DescriptorSet> mDescriptors {};
};