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

void DescriptorAllocator::InitPool(vk::Device device, uint32_t initialSets,
                                   std::span<PoolSizeRatio> poolRatios) {
    mRatios.clear();

    for (auto r : poolRatios) {
        mRatios.push_back(r);
    }

    auto newPool = CreatePool(device, initialSets, poolRatios);

    mSetsPerPool = initialSets * 1.5;

    mReadyPools.emplace_back(newPool);
}

void DescriptorAllocator::ClearDescriptors(vk::Device device) {
    for (auto p : mReadyPools) {
        device.resetDescriptorPool(p);
    }
    for (auto p : mFullPools) {
        device.resetDescriptorPool(p);
        mReadyPools.push_back(p);
    }
    mFullPools.clear();
}

void DescriptorAllocator::DestroyPool(vk::Device device) {
    for (auto p : mReadyPools) {
        device.destroy(p);
    }
    mReadyPools.clear();
    for (auto p : mFullPools) {
        device.destroy(p);
    }
    mFullPools.clear();
}

vk::DescriptorSet DescriptorAllocator::Allocate(vk::Device              device,
                                                vk::DescriptorSetLayout layout,
                                                void*                   pNext) {
    auto poolToUse = GetPool(device);

    vk::DescriptorSetAllocateInfo allocInfo {};
    allocInfo.setDescriptorPool(poolToUse).setSetLayouts(layout).setPNext(
        pNext);

    vk::DescriptorSet ds;
    vk::Result        result = device.allocateDescriptorSets(&allocInfo, &ds);

    if (result == vk::Result::eErrorOutOfPoolMemory
        || result == vk::Result::eErrorFragmentedPool) {

        mFullPools.push_back(poolToUse);

        poolToUse                = GetPool(device);
        allocInfo.descriptorPool = poolToUse;

        ds = device.allocateDescriptorSets(allocInfo)[0];
    }

    mReadyPools.push_back(poolToUse);
    return ds;
}

vk::DescriptorPool DescriptorAllocator::GetPool(vk::Device device) {
    vk::DescriptorPool newPool;
    if (!mReadyPools.empty()) {
        newPool = mReadyPools.back();
        mReadyPools.pop_back();
    } else {
        newPool = CreatePool(device, mSetsPerPool, mRatios);

        mSetsPerPool = mSetsPerPool * 1.5;
        if (mSetsPerPool > 4092) {
            mSetsPerPool = 4092;
        }
    }

    return newPool;
}

vk::DescriptorPool DescriptorAllocator::CreatePool(
    vk::Device device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.emplace_back(ratio.type,
                               static_cast<uint32_t>(ratio.ratio * setCount));
    }
    vk::DescriptorPoolCreateInfo poolInfo {};
    poolInfo.setMaxSets(setCount).setPoolSizes(poolSizes);

    return device.createDescriptorPool(poolInfo);
}

void DescriptorWriter::WriteImage(int                     binding,
                                  vk::DescriptorImageInfo imageInfo,
                                  vk::DescriptorType      type) {
    mImageInfos.push_back(imageInfo);

    vk::WriteDescriptorSet write {};
    write.setDstBinding(binding)
        .setDescriptorCount(1u)
        .setDescriptorType(type)
        .setImageInfo(imageInfo);

    mWrites.push_back(write);
}

void DescriptorWriter::WriteBuffer(int                      binding,
                                   vk::DescriptorBufferInfo bufferInfo,
                                   vk::DescriptorType       type) {
    mBufferInfos.push_back(bufferInfo);

    vk::WriteDescriptorSet write {};
    write.setDstBinding(binding)
        .setDescriptorCount(1u)
        .setDescriptorType(type)
        .setBufferInfo(bufferInfo);

    mWrites.push_back(write);
}

void DescriptorWriter::Clear() {
    mImageInfos.clear();
    mWrites.clear();
    mBufferInfos.clear();
}

void DescriptorWriter::UpdateSet(vk::Device device, vk::DescriptorSet set) {
    for (auto& write : mWrites) {
        write.setDstSet(set);
    }
    device.updateDescriptorSets(mWrites.size(), mWrites.data(), 0,
                                VK_NULL_HANDLE);
}