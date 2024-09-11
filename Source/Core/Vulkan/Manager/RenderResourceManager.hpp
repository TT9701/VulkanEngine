#pragma once

#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Native/RenderResource.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

struct DescriptorSetData {
    uint32_t binding;
    Type_STLString resource;
    vk::DescriptorType type;
    Type_STLString layoutBinding;
    ::std::optional<Type_STLString> imageView;
    ::std::optional<Sampler*> sampler;
};

class RenderResourceManager {
    using Type_RenderResources =
        Type_STLUnorderedMap_String<SharedPtr<RenderResource>>;

public:
    RenderResourceManager(Device* device, MemoryAllocator* allocator,
                          DescriptorManager* manager);

    RenderResource* CreateBuffer(const char* name, size_t size,
                                 vk::BufferUsageFlags usage,
                                 Buffer::MemoryType memType);

    RenderResource* CreateTexture(const char* name, RenderResource::Type type,
                                  vk::Format format, vk::Extent3D extent,
                                  vk::ImageUsageFlags usage,
                                  uint32_t mipLevels = 1,
                                  uint32_t arraySize = 1,
                                  uint32_t sampleCount = 1);

    RenderResource* operator[](const char* name) const;

    void CreateDescriptorSet(const char* name, uint32_t setIndex,
        const char* pipelineName, vk::ShaderStageFlags stage,
        Type_STLVector<DescriptorSetData> const& datas,
        uint32_t bufferIndex = 0);

private:
    Device* pDevice;
    MemoryAllocator* pAllocator;
    DescriptorManager* pDescManager;

    Type_RenderResources mResources {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core