#pragma once

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/RenderResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

class RenderResourceManager {
    using Type_RenderResources =
        Type_STLUnorderedMap_String<SharedPtr<RenderResource>>;

    using Fn_SizeRelation = ::std::function<vk::Extent2D(vk::Extent2D)>;

    using Type_Resource_SizeRelations =
        Type_STLUnorderedMap_String<Fn_SizeRelation>;

public:
    RenderResourceManager(VulkanContext& context);

    RenderResource& CreateBuffer(const char* name, size_t size,
                                 vk::BufferUsageFlags usage,
                                 Buffer::MemoryType memType,
                                 size_t texelSize = 1);

    RenderResource& CreateTexture(const char* name, RenderResource::Type type,
                                  vk::Format format, vk::Extent3D extent,
                                  vk::ImageUsageFlags usage,
                                  uint32_t mipLevels = 1,
                                  uint32_t arraySize = 1,
                                  uint32_t sampleCount = 1);

    RenderResource& CreateBuffer_ScreenSizeRelated(
        const char* name, size_t size, vk::BufferUsageFlags usage,
        Buffer::MemoryType memType, size_t texelSize = 1,
        Fn_SizeRelation fn = sFullSize);

    RenderResource& CreateTexture_ScreenSizeRelated(
        const char* name, RenderResource::Type type, vk::Format format,
        vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels = 1,
        uint32_t arraySize = 1, uint32_t sampleCount = 1,
        Fn_SizeRelation fn = sFullSize);

    RenderResource const& operator[](const char* name) const;

    void ResizeResources_ScreenSizeRelated(vk::Extent2D extent) const;

    Type_Resource_SizeRelations const&
    GetResources_SrcreenSizeRelated() const;

    Type_STLVector<Type_STLString> GetResourceNames_SrcreenSizeRelated() const;

private:
    static Fn_SizeRelation sFullSize;

private:
    VulkanContext& mContext;

    Type_RenderResources mResources {};
    Type_Resource_SizeRelations mScreenSizeRalatedResources {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core