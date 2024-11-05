#include "RenderResourceManager.hpp"

#include "Core/Vulkan/Native/Device.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResourceManager::Fn_SizeRelation RenderResourceManager::sFullSize {
    [](vk::Extent2D extent) {
        return extent;
    }};

RenderResourceManager::RenderResourceManager(Device* device,
                                             MemoryAllocator* allocator)
    : pDevice(device), pAllocator(allocator) {}

RenderResource* RenderResourceManager::CreateBuffer(const char* name,
                                                    size_t size,
                                                    vk::BufferUsageFlags usage,
                                                    Buffer::MemoryType memType,
                                                    size_t texelSize) {
    auto ptr = MakeShared<RenderResource>(pDevice, pAllocator,
                                          RenderResource::Type::Buffer, size,
                                          usage, memType, texelSize);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::CreateTexture(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount) {
    auto ptr =
        MakeShared<RenderResource>(pDevice, pAllocator, type, format, extent,
                                   usage, mipLevels, arraySize, sampleCount);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::CreateBuffer_ScreenSizeRelated(
    const char* name, size_t size, vk::BufferUsageFlags usage,
    Buffer::MemoryType memType, size_t texelSize, Fn_SizeRelation fn) {
    auto ptr = CreateBuffer(name, size, usage, memType, texelSize);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ptr;
}

RenderResource* RenderResourceManager::CreateTexture_ScreenSizeRelated(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount, Fn_SizeRelation fn) {
    auto ptr = CreateTexture(name, type, format, extent, usage, mipLevels,
                             arraySize, sampleCount);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ptr;
}

RenderResource* RenderResourceManager::operator[](const char* name) const {
    return mResources.at(name).get();
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

}  // namespace IntelliDesign_NS::Vulkan::Core