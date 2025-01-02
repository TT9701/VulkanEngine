#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace IntelliDesign_NS::Vulkan::Core {
class VulkanContext;

template <class Handle>
class Resource {
public:
    Resource(VulkanContext& context, Handle handle = VK_NULL_HANDLE);

    CLASS_NO_COPY(Resource);

    Resource(Resource&& other) noexcept;
    Resource& operator=(Resource&& other) noexcept;

    virtual ~Resource() = default;

    Type_STLString const& GetDebugName() const;

    Handle& GetHandle();

    Handle const& GetHandle() const;

    bool HasHandle() const;

    void SetName(const char* name);

    void SetHandle(Handle hdl);

private:
    VulkanContext& mContext;
    Handle mHandle;
    Type_STLString mDebugName;
};

template <class Handle>
Resource<Handle>::Resource(VulkanContext& context, Handle handle)
    : mContext(context), mHandle(handle) {}

template <class Handle>
Resource<Handle>::Resource(Resource&& other) noexcept
    : mContext(other.mContext), mHandle(std::exchange(other.mHandle, {})) {}

template <class Handle>
Resource<Handle>& Resource<Handle>::operator=(Resource&& other) noexcept {
    mContext = other.mContext;
    mHandle = std::exchange(other.mHandle, {});
    return *this;
}

template <class Handle>
Type_STLString const& Resource<Handle>::GetDebugName() const {
    return mDebugName;
}

template <class Handle>
Handle& Resource<Handle>::GetHandle() {
    return mHandle;
}

template <class Handle>
Handle const& Resource<Handle>::GetHandle() const {
    return mHandle;
}

template <class Handle>
bool Resource<Handle>::HasHandle() const {
    return mHandle != VK_NULL_HANDLE;
}

template <class Handle>
void Resource<Handle>::SetName(const char* name) {
    mDebugName = name;
    if (!mDebugName.empty()) {
        mContext.SetName(mHandle, mDebugName.c_str());
    }
}

template <class Handle>
void Resource<Handle>::SetHandle(Handle hdl) {
    mHandle = hdl;
}

template <class BuilderType, class CreateInfoType>
class ResourceBuilder {
public:
    ResourceBuilder(const ResourceBuilder& other) = delete;

    VmaAllocationCreateInfo const& GetAllocationCreateInfo() const;

    CreateInfoType const& GetCreateInfo() const;

    Type_STLString const& GetName() const;

    BuilderType& SetDebugName(const char* name);

    BuilderType& SetImplicitSharingMode();

    BuilderType& SetMemoryTypeBits(uint32_t typeBits);

    BuilderType& SetQueueFamilies(uint32_t count,
                                  const uint32_t* familyIndices);

    BuilderType& SetQueueFamilies(std::vector<uint32_t> const& queueFamilies);

    BuilderType& SetSharingMode(vk::SharingMode sharingMode);

    BuilderType& SetVmaFlags(VmaAllocationCreateFlags flags);

    BuilderType& SetVmaPool(VmaPool pool);

    BuilderType& SetVmaPreferredFlags(vk::MemoryPropertyFlags flags);

    BuilderType& SetVmaRequiredFlags(vk::MemoryPropertyFlags flags);

    BuilderType& SetVmaUsage(VmaMemoryUsage usage);

protected:
    ResourceBuilder(const CreateInfoType& createInfo);

    CreateInfoType& GetCreateInfo();

protected:
    VmaAllocationCreateInfo mAllocCreateInfo {};
    CreateInfoType mCreateInfo {};
    Type_STLString mDebugName {};
};

template <class BuilderType, class CreateInfoType>
VmaAllocationCreateInfo const&
ResourceBuilder<BuilderType, CreateInfoType>::GetAllocationCreateInfo() const {
    return mAllocCreateInfo;
}

template <class BuilderType, class CreateInfoType>
CreateInfoType const&
ResourceBuilder<BuilderType, CreateInfoType>::GetCreateInfo() const {
    return mCreateInfo;
}

template <class BuilderType, class CreateInfoType>
Type_STLString const& ResourceBuilder<BuilderType, CreateInfoType>::GetName()
    const {
    return mDebugName;
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetDebugName(
    const char* name) {
    mDebugName = name;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType&
ResourceBuilder<BuilderType, CreateInfoType>::SetImplicitSharingMode() {
    mCreateInfo.sharingMode = (1 < mCreateInfo.queueFamilyIndexCount)
                                ? vk::SharingMode::eConcurrent
                                : vk::SharingMode::eExclusive;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetMemoryTypeBits(
    uint32_t typeBits) {
    mAllocCreateInfo.memoryTypeBits = typeBits;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetQueueFamilies(
    uint32_t count, const uint32_t* familyIndices) {
    mCreateInfo.queueFamilyIndexCount = count;
    mCreateInfo.pQueueFamilyIndices = familyIndices;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetQueueFamilies(
    std::vector<uint32_t> const& queueFamilies) {
    return SetQueueFamilies(static_cast<uint32_t>(queueFamilies.size()),
                            queueFamilies.data());
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetSharingMode(
    vk::SharingMode sharingMode) {
    mCreateInfo.sharingMode = sharingMode;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetVmaFlags(
    VmaAllocationCreateFlags flags) {
    mAllocCreateInfo.flags = flags;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetVmaPool(
    VmaPool pool) {
    mAllocCreateInfo.pool = pool;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetVmaPreferredFlags(
    vk::MemoryPropertyFlags flags) {
    mAllocCreateInfo.preferredFlags = static_cast<VkMemoryPropertyFlags>(flags);
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetVmaRequiredFlags(
    vk::MemoryPropertyFlags flags) {
    mAllocCreateInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(flags);
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
BuilderType& ResourceBuilder<BuilderType, CreateInfoType>::SetVmaUsage(
    VmaMemoryUsage usage) {
    mAllocCreateInfo.usage = usage;
    return *static_cast<BuilderType*>(this);
}

template <class BuilderType, class CreateInfoType>
ResourceBuilder<BuilderType, CreateInfoType>::ResourceBuilder(
    const CreateInfoType& createInfo)
    : mCreateInfo(createInfo) {
    mAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
}

template <class BuilderType, class CreateInfoType>
CreateInfoType& ResourceBuilder<BuilderType, CreateInfoType>::GetCreateInfo() {
    return mCreateInfo;
}

}  // namespace IntelliDesign_NS::Vulkan::Core