#include "Descriptors.hpp"

#include "Context.hpp"
#include "VulkanHelper.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

namespace __Detail {

void SetLayoutBuilder::AddBinding(uint32_t binding, uint32_t descCount,
                                  vk::DescriptorType type) {
    vk::DescriptorSetLayoutBinding newbind {};
    newbind.setBinding(binding).setDescriptorCount(descCount).setDescriptorType(
        type);

    mBindings.push_back(newbind);
}

void SetLayoutBuilder::Clear() {
    mBindings.clear();
}

vk::DescriptorSetLayout SetLayoutBuilder::Build(
    Context* context, vk::ShaderStageFlags shaderStages,
    vk::DescriptorSetLayoutCreateFlags flags, void* pNext) {
    for (auto& b : mBindings) {
        b.stageFlags |= shaderStages;
    }

    vk::DescriptorSetLayoutCreateInfo info {};
    info.setPNext(pNext).setBindings(mBindings).setFlags(flags);
    return context->GetDeviceHandle().createDescriptorSetLayout(info);
}

void DescriptorAllocator::InitPool(Context* context, uint32_t initialSets,
                                   std::span<DescPoolSizeRatio> poolRatios) {
    mRatios.clear();

    for (auto r : poolRatios) {
        mRatios.push_back(r);
    }

    auto newPool = CreatePool(context, initialSets, poolRatios);

    mSetsPerPool =
        static_cast<uint32_t>(static_cast<double>(initialSets) * 1.5);

    mReadyPools.emplace_back(newPool);
}

void DescriptorAllocator::ClearDescriptors(Context* context) {
    for (auto p : mReadyPools) {
        context->GetDeviceHandle().resetDescriptorPool(p);
    }
    for (auto p : mFullPools) {
        context->GetDeviceHandle().resetDescriptorPool(p);
        mReadyPools.push_back(p);
    }
    mFullPools.clear();
}

void DescriptorAllocator::DestroyPool(Context* context) {
    for (auto p : mReadyPools) {
        context->GetDeviceHandle().destroy(p);
    }
    mReadyPools.clear();
    for (auto p : mFullPools) {
        context->GetDeviceHandle().destroy(p);
    }
    mFullPools.clear();
}

vk::DescriptorSet DescriptorAllocator::Allocate(Context* context,
                                                vk::DescriptorSetLayout layout,
                                                void* pNext) {
    auto poolToUse = GetPool(context);

    vk::DescriptorSetAllocateInfo allocInfo {};
    allocInfo.setDescriptorPool(poolToUse).setSetLayouts(layout).setPNext(
        pNext);

    vk::DescriptorSet ds;

    vk::Result result =
        context->GetDeviceHandle().allocateDescriptorSets(&allocInfo, &ds);

    if (result == vk::Result::eErrorOutOfPoolMemory
        || result == vk::Result::eErrorFragmentedPool) {

        mFullPools.push_back(poolToUse);

        poolToUse = GetPool(context);
        allocInfo.descriptorPool = poolToUse;

        ds = context->GetDeviceHandle().allocateDescriptorSets(allocInfo)[0];
    }

    mReadyPools.push_back(poolToUse);
    return ds;
}

vk::DescriptorPool DescriptorAllocator::GetPool(Context* context) {
    vk::DescriptorPool newPool;
    if (!mReadyPools.empty()) {
        newPool = mReadyPools.back();
        mReadyPools.pop_back();
    } else {
        newPool = CreatePool(context, mSetsPerPool, mRatios);

        mSetsPerPool =
            static_cast<uint32_t>(static_cast<double>(mSetsPerPool) * 1.5);
        if (mSetsPerPool > 4092) {
            mSetsPerPool = 4092;
        }
    }

    return newPool;
}

vk::DescriptorPool DescriptorAllocator::CreatePool(
    Context* context, uint32_t setCount,
    std::span<DescPoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (auto& ratio : poolRatios) {
        poolSizes.emplace_back(ratio.mType,
                               static_cast<uint32_t>(ratio.mRatio * setCount));
    }
    vk::DescriptorPoolCreateInfo poolInfo {};
    poolInfo.setMaxSets(setCount).setPoolSizes(poolSizes);

    return context->GetDeviceHandle().createDescriptorPool(poolInfo);
}

void DescriptorWriter::WriteImage(int binding,
                                  vk::DescriptorImageInfo const& imageInfo,
                                  vk::DescriptorType type) {
    mImageInfos.push_back(imageInfo);

    vk::WriteDescriptorSet write {};
    write.setDstBinding(binding)
        .setDescriptorCount(1u)
        .setDescriptorType(type)
        .setImageInfo(mImageInfos.back());

    mWrites.push_back(write);
}

void DescriptorWriter::WriteBuffer(int binding,
                                   vk::DescriptorBufferInfo const& bufferInfo,
                                   vk::DescriptorType type) {
    mBufferInfos.push_back(bufferInfo);

    vk::WriteDescriptorSet write {};
    write.setDstBinding(binding)
        .setDescriptorCount(1u)
        .setDescriptorType(type)
        .setBufferInfo(mBufferInfos.back());

    mWrites.push_back(write);
}

void DescriptorWriter::Clear() {
    mImageInfos.clear();
    mWrites.clear();
    mBufferInfos.clear();
}

void DescriptorWriter::UpdateSet(Context* context,
                                 vk::DescriptorSet set) {
    for (auto& write : mWrites) {
        write.setDstSet(set);
    }
    context->GetDeviceHandle().updateDescriptorSets(mWrites, {});

    Clear();
}

}  // namespace VulkanCore::__Detail

DescriptorManager::DescriptorManager(
    Context* context, uint32_t initialSets,
    ::std::span<DescPoolSizeRatio> poolRatio)
    : pContext(context) {
    mDescAllocator.InitPool(context, initialSets, poolRatio);
}

DescriptorManager::~DescriptorManager() {
    for (auto& layout : mSetLayouts) {
        pContext->GetDeviceHandle().destroy(layout.second);
    }

    mDescAllocator.DestroyPool(pContext);
}

void DescriptorManager::AddDescSetLayoutBinding(uint32_t binding,
                                                      uint32_t descCount,
                                                      vk::DescriptorType type) {
    mSetLayoutBuilder.AddBinding(binding, descCount, type);
}

vk::DescriptorSetLayout DescriptorManager::BuildDescSetLayout(
    ::std::string const& name, vk::ShaderStageFlags shaderStages,
    vk::DescriptorSetLayoutCreateFlags flags, void* pNext) {
    const auto layout =
        mSetLayoutBuilder.Build(pContext, shaderStages, flags, pNext);

    mSetLayouts.emplace(name, layout);

    mSetLayoutBuilder.Clear();

    return layout;
}

vk::DescriptorSetLayout DescriptorManager::GetDescSetLayout(
    std::string const& name) const {
    return mSetLayouts.at(name);
}

void DescriptorManager::ClearSetLayout() {
    mSetLayoutBuilder.Clear();
}

vk::DescriptorSet DescriptorManager::Allocate(
    ::std::string const& name, vk::DescriptorSetLayout layout, void* pNext) {
    const auto desc = mDescAllocator.Allocate(pContext, layout, pNext);

    mDescriptors.emplace(name, desc);

    return desc;
}

vk::DescriptorSet DescriptorManager::GetDescriptor(
    std::string const& name) const {
    return mDescriptors.at(name);
}

void DescriptorManager::ClearDescriptors() {
    mDescAllocator.ClearDescriptors(pContext);
}

void DescriptorManager::WriteImage(int binding,
                                         vk::DescriptorImageInfo imageInfo,
                                         vk::DescriptorType type) {
    mDescWriter.WriteImage(binding, imageInfo, type);
}

void DescriptorManager::WriteBuffer(int binding,
                                          vk::DescriptorBufferInfo bufferInfo,
                                          vk::DescriptorType type) {
    mDescWriter.WriteBuffer(binding, bufferInfo, type);
}

void DescriptorManager::Clear() {
    mDescWriter.Clear();
}

void DescriptorManager::UpdateSet(vk::DescriptorSet set) {
    mDescWriter.UpdateSet(pContext, set);
}

}  // namespace IntelliDesign_NS::Vulkan::Core