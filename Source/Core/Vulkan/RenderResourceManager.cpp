#include "RenderResourceManager.hpp"

#include "Device.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResourceManager::RenderResourceManager(Device* device,
                                             MemoryAllocator* allocator)
    : pDevice(device), pAllocator(allocator) {}

RenderResource* RenderResourceManager::CreateBuffer(
    std::string_view name, size_t size, vk::BufferUsageFlags usage,
    Buffer::MemoryType memType) {
    auto ptr = MakeShared<RenderResource>(pDevice, pAllocator,
                                          RenderResource::Type::Buffer, size,
                                          usage, memType);

    pDevice->SetObjectName(ptr->GetBufferHandle(),
                           static_cast<::std::string>(name).c_str());

    mResources.emplace(static_cast<::std::string>(name), ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::CreateTexture(
    std::string_view name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount) {
    auto ptr =
        MakeShared<RenderResource>(pDevice, pAllocator, type, format, extent,
                                   usage, mipLevels, arraySize, sampleCount);

    pDevice->SetObjectName(ptr->GetTexHandle(),
                           static_cast<::std::string>(name).c_str());

    mResources.emplace(static_cast<::std::string>(name), ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::GetResource(
    std::string_view name) const {
    return mResources.at(static_cast<::std::string>(name)).get();
}

}  // namespace IntelliDesign_NS::Vulkan::Core