#pragma once

#include "MemoryAllocator.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Image;
class VulkanContext;

using ImagePtr = UniquePtr<Image>;

class ImageBuilder : public ResourceBuilder<ImageBuilder, vk::ImageCreateInfo> {
    using Parent = ResourceBuilder<ImageBuilder, vk::ImageCreateInfo>;

    ImageBuilder(vk::Extent3D const& extent);

public:
    ImageBuilder& SetFormat(vk::Format format);

    ImageBuilder& SetImageType(vk::ImageType type);

    ImageBuilder& SetArrayLayers(uint32_t layers);

    ImageBuilder& SetMipLevels(uint32_t levels);

    ImageBuilder& SetSampleCount(vk::SampleCountFlagBits sampleCount);

    ImageBuilder& SetTiling(vk::ImageTiling tiling);

    ImageBuilder& SetUsage(vk::ImageUsageFlags usage);

    ImageBuilder& SetFlags(vk::ImageCreateFlags flags);

    Image Build(VulkanContext& context) const;

    ImagePtr BuildUnique(VulkanContext& context) const;

    friend class Image;
};

class Image : public Allocated<vk::Image> {
public:
    Image(VulkanContext& context, vk::Image handle, const vk::Extent3D& extent,
          vk::Format format, vk::ImageUsageFlags image_usage,
          vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    Image(VulkanContext& context, ImageBuilder const& builder);

    ~Image() override;

    static ImageBuilder GetBuilder(vk::Extent3D const& extent);
    static ImageBuilder GetBuilder(vk::Extent2D const& extent);
    static ImageBuilder GetBuilder(uint32_t width, uint32_t height = 1,
                                   uint32_t depth = 1);

    CLASS_NO_COPY(Image);

    Image(Image&& other) noexcept;

    Image& operator=(Image&&) = delete;

    uint8_t* Map();

    vk::ImageType GetType() const;

    const vk::Extent3D& GetExtent() const;

    vk::Format GetFormat() const;

    vk::SampleCountFlagBits GetSampleCount() const;

    vk::ImageUsageFlags GetUsage() const;

    vk::ImageTiling GetTiling() const;

    vk::ImageSubresource GetSubresource() const;

    uint32_t GetArrayLayerCount() const;

    // std::unordered_set<ImageView*>& get_views();

private:
    vk::ImageCreateInfo mCreateInfo;
    vk::ImageSubresource mSubresource;
    // std::unordered_set<vkb::core::HPPImageView*> views;
};

}  // namespace IntelliDesign_NS::Vulkan::Core