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

    vk::DescriptorPool mPool;

    void InitPool(vk::Device device, uint32_t maxSets,
                  ::std::span<PoolSizeRatio> poolRatios);
    void ClearDescriptors(vk::Device device);
    void DestroyPool(vk::Device device);

    ::std::vector<vk::DescriptorSet> Allocate(vk::Device device, vk::DescriptorSetLayout layout);
};