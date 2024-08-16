#pragma once

#include "Core/Utilities/MemoryPool.hpp"
#include "RenderResource.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderResourceManager {
public:
    RenderResourceManager(Device* device, MemoryAllocator* allocator);

    RenderResource* CreateBuffer(::std::string_view name, size_t size,
                                 vk::BufferUsageFlags usage,
                                 Buffer::MemoryType memType);

    RenderResource* CreateTexture(
        ::std::string_view name, RenderResource::Type type, vk::Format format,
        vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels = 1,
        uint32_t arraySize = 1, uint32_t sampleCount = 1);

    RenderResource* GetResource(::std::string_view name) const;

private:
    Device* pDevice;
    MemoryAllocator* pAllocator;

    ::std::unordered_map<::std::string, SharedPtr<RenderResource>>
        mResources {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core