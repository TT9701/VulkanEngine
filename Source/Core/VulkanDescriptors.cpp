#include "VulkanDescriptors.hpp"

#include "VulkanHelper.hpp"

void DescriptorLayoutBuilder::AddBinding(uint32_t           binding,
                                          vk::DescriptorType type) {
    vk::DescriptorSetLayoutBinding newbind {};
    newbind.setBinding(binding).setDescriptorCount(1u).setDescriptorType(type);

    mBindings.push_back(newbind);
}

void DescriptorLayoutBuilder::Clear() {
    mBindings.clear();
}

vk::DescriptorSetLayout DescriptorLayoutBuilder::Build(
    vk::Device device, vk::ShaderStageFlags shaderStages, void* pNext,
    vk::DescriptorSetLayoutCreateFlags flags) {
    for (auto& b : mBindings) {
        b.stageFlags |= shaderStages;
    }
    vk::DescriptorSetLayoutCreateInfo info {};
    info.setPNext(pNext).setBindings(mBindings).setFlags(flags);

    return device.createDescriptorSetLayout(info);
}

void DescriptorAllocator::InitPool(vk::Device device, uint32_t maxSets,
                                   std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.push_back(vk::DescriptorPoolSize {
            ratio.type, static_cast<uint32_t>(ratio.ratio * maxSets)});
    }

    vk::DescriptorPoolCreateInfo poolInfo {};
    poolInfo.setMaxSets(maxSets).setPoolSizes(poolSizes);

    mPool = device.createDescriptorPool(poolInfo);
}

void DescriptorAllocator::ClearDescriptors(vk::Device device) {
    device.resetDescriptorPool(mPool);
}

void DescriptorAllocator::DestroyPool(vk::Device device) {
    device.destroy(mPool);
}

::std::vector<vk::DescriptorSet> DescriptorAllocator::Allocate(
    vk::Device device, vk::DescriptorSetLayout layout) {
    vk::DescriptorSetAllocateInfo allocInfo {};
    allocInfo.setDescriptorPool(mPool).setSetLayouts(layout);

    return device.allocateDescriptorSets(allocInfo);
}