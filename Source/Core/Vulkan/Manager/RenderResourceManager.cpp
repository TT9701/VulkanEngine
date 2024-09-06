#include "RenderResourceManager.hpp"

#include "Core/Vulkan/Native/Device.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResourceManager::RenderResourceManager(Device* device,
                                             MemoryAllocator* allocator)
    : pDevice(device), pAllocator(allocator) {}

RenderResource* RenderResourceManager::CreateBuffer(
    const char* name, size_t size, vk::BufferUsageFlags usage,
    Buffer::MemoryType memType) {
    auto ptr = MakeShared<RenderResource>(pDevice, pAllocator,
                                          RenderResource::Type::Buffer, size,
                                          usage, memType);

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

RenderResource* RenderResourceManager::GetResource(const char* name) const {
    return mResources.at(name).get();
}

}  // namespace IntelliDesign_NS::Vulkan::Core