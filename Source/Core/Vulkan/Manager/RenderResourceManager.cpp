#include "RenderResourceManager.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResourceManager::Fn_SizeRelation RenderResourceManager::sFullSize {
    [](vk::Extent2D extent) {
        return extent;
    }};

RenderResourceManager::RenderResourceManager(VulkanContext& context)
    : mContext(context) {}

RenderResource& RenderResourceManager::CreateBuffer(const char* name,
                                                    size_t size,
                                                    vk::BufferUsageFlags usage,
                                                    Buffer::MemoryType memType,
                                                    size_t texelSize) {
    auto ptr =
        MakeShared<RenderResource>(mContext, RenderResource::Type::Buffer, size,
                                   usage, memType, texelSize);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return *ptr;
}

RenderResource& RenderResourceManager::CreateTexture(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount) {
    auto ptr = MakeShared<RenderResource>(mContext, type, format, extent, usage,
                                          mipLevels, arraySize, sampleCount);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return *ptr;
}

RenderResource& RenderResourceManager::CreateBuffer_ScreenSizeRelated(
    const char* name, size_t size, vk::BufferUsageFlags usage,
    Buffer::MemoryType memType, size_t texelSize, Fn_SizeRelation fn) {
    auto& ref = CreateBuffer(name, size, usage, memType, texelSize);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ref;
}

RenderResource& RenderResourceManager::CreateTexture_ScreenSizeRelated(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount, Fn_SizeRelation fn) {
    auto& ref = CreateTexture(name, type, format, extent, usage, mipLevels,
                             arraySize, sampleCount);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ref;
}

RenderResource const& RenderResourceManager::operator[](const char* name) const {
    return *mResources.at(name);
}

void RenderResourceManager::ResizeResources_ScreenSizeRelated(
    vk::Extent2D extent) const {
    for (auto& screenSizeResource : mScreenSizeRalatedResources) {
        auto fn = screenSizeResource.second;
        auto newExtent = fn(extent);
        mResources.at(screenSizeResource.first)->Resize(newExtent);
    }
}

RenderResourceManager::Type_Resource_SizeRelations const&
RenderResourceManager::GetResources_SrcreenSizeRelated() const {
    return mScreenSizeRalatedResources;
}

Type_STLVector<Type_STLString>
RenderResourceManager::GetResourceNames_SrcreenSizeRelated() const {
    Type_STLVector<Type_STLString> names {};
    for (auto const& [name, _] : mScreenSizeRalatedResources) {
        names.push_back(name);
    }
    return names;
}

}  // namespace IntelliDesign_NS::Vulkan::Core