#pragma once

#include "Utilities/VulkanUtilities.hpp"

class DescriptorLayoutBuilder {
public:
    std::vector<vk::DescriptorSetLayoutBinding> mBindings;

    void AddBinding(uint32_t binding, vk::DescriptorType type);
    void Clear();
    vk::DescriptorSetLayout Build(
        vk::Device device, vk::ShaderStageFlags shaderStages,
        void* pNext = nullptr, vk::DescriptorSetLayoutCreateFlags flags = {});
};

class DescriptorAllocator {
public:
    struct PoolSizeRatio {
        vk::DescriptorType type;
        float              ratio;
    };

    void InitPool(vk::Device device, uint32_t initialSets,
                  ::std::span<PoolSizeRatio> poolRatios);
    void ClearDescriptors(vk::Device device);
    void DestroyPool(vk::Device device);

    vk::DescriptorSet Allocate(vk::Device              device,
                               vk::DescriptorSetLayout layout,
                               void*                   pNext = nullptr);

private:
    vk::DescriptorPool GetPool(vk::Device device);

    vk::DescriptorPool CreatePool(vk::Device device, uint32_t setCount,
                                  std::span<PoolSizeRatio> poolRatios);

    ::std::vector<PoolSizeRatio>      mRatios {};
    ::std::vector<vk::DescriptorPool> mFullPools {};
    ::std::vector<vk::DescriptorPool> mReadyPools {};
    uint32_t                          mSetsPerPool {};
};

class DescriptorWriter {
public:
    ::std::deque<vk::DescriptorImageInfo>  mImageInfos {};
    ::std::deque<vk::DescriptorBufferInfo> mBufferInfos {};
    ::std::vector<vk::WriteDescriptorSet>  mWrites {};

    void WriteImage(int binding, vk::DescriptorImageInfo imageInfo,
                    vk::DescriptorType type);

    void WriteBuffer(int binding, vk::DescriptorBufferInfo bufferInfo,
                     vk::DescriptorType type);

    void Clear();

    void UpdateSet(vk::Device device, vk::DescriptorSet set);
};