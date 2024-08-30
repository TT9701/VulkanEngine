#pragma once

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Native/RenderResource.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderResourceManager {
    using Type_RenderResources =
        Type_STLUnorderedMap_String<SharedPtr<RenderResource>>;

public:
    RenderResourceManager(Device* device, MemoryAllocator* allocator);

    RenderResource* CreateBuffer(const char* name, size_t size,
                                 vk::BufferUsageFlags usage,
                                 Buffer::MemoryType memType);

    RenderResource* CreateTexture(const char* name, RenderResource::Type type,
                                  vk::Format format, vk::Extent3D extent,
                                  vk::ImageUsageFlags usage,
                                  uint32_t mipLevels = 1,
                                  uint32_t arraySize = 1,
                                  uint32_t sampleCount = 1);

    RenderResource* GetResource(const char* name) const;

private:
    Device* pDevice;
    MemoryAllocator* pAllocator;

    Type_RenderResources mResources {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core