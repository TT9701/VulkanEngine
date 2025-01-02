#include "Image.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

namespace {

inline vk::ImageType FindImageType(vk::Extent3D const& extent) {
    uint32_t dimNum = !!extent.width + !!extent.height + (1 < extent.depth);
    switch (dimNum) {
        case 1: return vk::ImageType::e1D;
        case 2: return vk::ImageType::e2D;
        case 3: return vk::ImageType::e3D;
        default:
            throw std::runtime_error("No image type found.");
            return vk::ImageType();
    }
}

}  // namespace

ImageBuilder::ImageBuilder(vk::Extent3D const& extent)
    : Parent(vk::ImageCreateInfo {
          {}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, extent, 1, 1}) {}

ImageBuilder& ImageBuilder::SetFormat(vk::Format format) {
    mCreateInfo.format = format;
    return *this;
}

ImageBuilder& ImageBuilder::SetImageType(vk::ImageType type) {
    mCreateInfo.imageType = type;
    return *this;
}

ImageBuilder& ImageBuilder::SetArrayLayers(uint32_t layers) {
    mCreateInfo.arrayLayers = layers;
    return *this;
}

ImageBuilder& ImageBuilder::SetMipLevels(uint32_t levels) {
    mCreateInfo.mipLevels = levels;
    return *this;
}

ImageBuilder& ImageBuilder::SetSampleCount(
    vk::SampleCountFlagBits sampleCount) {
    mCreateInfo.samples = sampleCount;
    return *this;
}

ImageBuilder& ImageBuilder::SetTiling(vk::ImageTiling tiling) {
    mCreateInfo.tiling = tiling;
    return *this;
}

ImageBuilder& ImageBuilder::SetUsage(vk::ImageUsageFlags usage) {
    mCreateInfo.usage = usage;
    return *this;
}

ImageBuilder& ImageBuilder::SetFlags(vk::ImageCreateFlags flags) {
    mCreateInfo.flags = flags;
    return *this;
}

Image ImageBuilder::Build(VulkanContext& context) const {
    return Image(context, *this);
}

ImagePtr ImageBuilder::BuildUnique(VulkanContext& context) const {
    return MakeUnique<Image>(context, *this);
}

Image::Image(VulkanContext& context, vk::Image handle,
             const vk::Extent3D& extent, vk::Format format,
             vk::ImageUsageFlags image_usage,
             vk::SampleCountFlagBits sampleCount)
    : Allocated {context, handle} {
    mCreateInfo.samples = sampleCount;
    mCreateInfo.format = format;
    mCreateInfo.extent = extent;
    mCreateInfo.imageType = FindImageType(extent);
    mCreateInfo.arrayLayers = 1;
    mCreateInfo.mipLevels = 1;
    mSubresource.mipLevel = 1;
    mSubresource.arrayLayer = 1;
}

Image::Image(VulkanContext& context, ImageBuilder const& builder)
    : Allocated {context, context.GetVmaAllocator(),
                 builder.GetAllocationCreateInfo(), VK_NULL_HANDLE},
      mCreateInfo {builder.GetCreateInfo()} {
    GetHandle() = CreateImage(mCreateInfo.operator const VkImageCreateInfo&());
    mSubresource.arrayLayer = mCreateInfo.arrayLayers;
    mSubresource.mipLevel = mCreateInfo.mipLevels;
    if (!builder.GetName().empty()) {
        SetName(builder.GetName().c_str());
    }
}

Image::~Image() {
    DestroyImage(GetHandle());
}

ImageBuilder Image::GetBuilder(vk::Extent3D const& extent) {
    return {extent};
}

ImageBuilder Image::GetBuilder(vk::Extent2D const& extent) {
    return {vk::Extent3D {extent.width, extent.height, 1}};
}

ImageBuilder Image::GetBuilder(uint32_t width, uint32_t height,
                               uint32_t depth) {
    return {vk::Extent3D {width, height, depth}};
}

Image::Image(Image&& other) noexcept
    : Allocated {std::move(other)},
      mCreateInfo(std::exchange(other.mCreateInfo, {})),
      mSubresource(std::exchange(other.mSubresource, {})) /*,
      views(std::exchange(other.views, {}))*/
{}

uint8_t* Image::Map() {
    if (mCreateInfo.tiling != vk::ImageTiling::eLinear) {
        DBG_LOG_INFO("Mapping image memory that is not linear");
    }
    return Allocated::Map();
}

vk::ImageType Image::GetType() const {
    return mCreateInfo.imageType;
}

const vk::Extent3D& Image::GetExtent() const {
    return mCreateInfo.extent;
}

vk::Format Image::GetFormat() const {
    return mCreateInfo.format;
}

vk::SampleCountFlagBits Image::GetSampleCount() const {
    return mCreateInfo.samples;
}

vk::ImageUsageFlags Image::GetUsage() const {
    return mCreateInfo.usage;
}

vk::ImageTiling Image::GetTiling() const {
    return mCreateInfo.tiling;
}

vk::ImageSubresource Image::GetSubresource() const {
    return mSubresource;
}

uint32_t Image::GetArrayLayerCount() const {
    return mCreateInfo.arrayLayers;
}
}  // namespace IntelliDesign_NS::Vulkan::Core