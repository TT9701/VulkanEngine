#include "Texture.hpp"

#include "Device.hpp"
#include "MemoryAllocator.hpp"
#include "VulkanHelper.hpp"

//
// void Texture::CopyToImage(vk::CommandBuffer cmd, vk::Image dstImage,
//                                 vk::Extent2D srcExtent,
//                                 vk::Extent2D dstExtent) {
//     vk::ImageBlit2 blitRegion {};
//     blitRegion
//         .setSrcOffsets(
//             {vk::Offset3D {},
//              vk::Offset3D {static_cast<int32_t>(srcExtent.width),
//                            static_cast<int32_t>(srcExtent.height), 1}})
//         .setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
//         .setDstOffsets(
//             {vk::Offset3D {},
//              vk::Offset3D {static_cast<int32_t>(dstExtent.width),
//                            static_cast<int32_t>(dstExtent.height), 1}})
//         .setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
//
//     vk::BlitImageInfo2 blitInfo {};
//     blitInfo.setDstImage(dstImage)
//         .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
//         .setSrcImage(mImage)
//         .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
//         .setFilter(vk::Filter::eLinear)
//         .setRegions(blitRegion);
//
//     cmd.blitImage2(blitInfo);
// }

namespace IntelliDesign_NS::Vulkan::Core {

Texture::Texture(Device* device,
                             MemoryAllocator* allocator, Type type,
                             vk::Format format, vk::Extent3D extent,
                             vk::ImageUsageFlags usage, uint32_t mipLevels,
                             uint32_t arraySize, uint32_t sampleCount)
    : pDevice(device),
      pAllocator(allocator),
      bOwnsImage(true),
      mType(type),
      mFormat(format),
      mExtent3D(extent),
      mUsageFlags(usage),
      mMipLevels(mipLevels),
      mArraySize(arraySize),
      mSampleCount(sampleCount),
      mHandle(CreateImage()) {}

Texture::Texture(Device* device, vk::Image handle, Type type,
                             vk::Format format, vk::Extent3D extent,
                             uint32_t arraySize, uint32_t sampleCount)
    : pDevice(device),
      pAllocator(nullptr),
      bOwnsImage(false),
      mType(type),
      mFormat(format),
      mExtent3D(extent),
      mMipLevels(1),
      mArraySize(arraySize),
      mSampleCount(sampleCount),
      mHandle(handle) {}

Texture::~Texture() {
    if (bOwnsImage)
        vmaDestroyImage(pAllocator->GetHandle(), mHandle, mAllocation);

    for (auto& [_, view] : mViews) {
        pDevice->GetHandle().destroy(view);
    }
}

void Texture::CreateImageView(std::string_view name,
                                    vk::ImageAspectFlags aspect,
                                    uint32_t mostDetailedMip, uint32_t mipCount,
                                    uint32_t firstArray, uint32_t arraySize) {
    vk::ImageSubresourceRange range {};
    range.setAspectMask(aspect)
        .setBaseMipLevel(mostDetailedMip)
        .setLevelCount(mipCount)
        .setBaseArrayLayer(firstArray)
        .setLayerCount(arraySize);

    vk::ImageViewType type {};
    switch (mType) {
        case Type::Texture1D:
            if (mArraySize > 1) {
                type = vk::ImageViewType::e1DArray;
            } else if (mArraySize == 1) {
                type = vk::ImageViewType::e1D;
            }
            break;
        case Type::Texture2D:
            if (mArraySize > 1) {
                type = vk::ImageViewType::e2DArray;
            } else if (mArraySize == 1) {
                type = vk::ImageViewType::e2D;
            }
            break;
        case Type::Texture3D: type = vk::ImageViewType::e3D; break;
        case Type::TextureCube:
            if (mArraySize > 1) {
                type = vk::ImageViewType::eCubeArray;
            } else if (mArraySize == 1) {
                type = vk::ImageViewType::eCube;
            }
            break;
        case Type::Texture2DMultisample: type = vk::ImageViewType::e2D; break;
        default: throw ::std::runtime_error("ERROR: Invalid Texture Type.");
    }

    vk::ImageViewCreateInfo info {};
    info.setImage(mHandle)
        .setFormat(mFormat)
        .setSubresourceRange(range)
        .setViewType(type);

    auto view = pDevice->GetHandle().createImageView(info);

    pDevice->SetObjectName(view, static_cast<::std::string>(name).c_str());

    mViews.emplace(static_cast<::std::string>(name), view);
}

uint32_t Texture::GetWidth(uint32_t mipLevel) const {
    return (mipLevel == 0) || (mipLevel < mMipLevels)
             ? std::max(1U, mExtent3D.width >> mipLevel)
             : 0;
}

uint32_t Texture::GetHeight(uint32_t mipLevel) const {
    return (mipLevel == 0) || (mipLevel < mMipLevels)
             ? std::max(1U, mExtent3D.height >> mipLevel)
             : 0;
}

uint32_t Texture::GetDepth(uint32_t mipLevel) const {
    return (mipLevel == 0) || (mipLevel < mMipLevels)
             ? std::max(1U, mExtent3D.depth >> mipLevel)
             : 0;
}

uint32_t Texture::GetMipCount() const {
    return mMipLevels;
}

uint32_t Texture::GetSampleCount() const {
    return mSampleCount;
}

uint32_t Texture::GetArraySize() const {
    return mArraySize;
}

vk::Format Texture::GetFormat() const {
    return mFormat;
}

vk::Image Texture::GetHandle() const {
    return mHandle;
}

vk::ImageView Texture::GetViewHandle(::std::string_view name) const {
    return mViews.at(static_cast<::std::string>(name));
}

vk::Image Texture::CreateImage() {
    VkImage image {};

    vk::ImageType type {};
    switch (mType) {
        case Type::Texture1D:
            VE_ASSERT(mExtent3D.height == 1 && mExtent3D.depth == 1
                          && mSampleCount == 1,
                      "");
            type = vk::ImageType::e1D;
            break;
        case Type::Texture2D:
            VE_ASSERT(mExtent3D.depth == 1 && mSampleCount == 1, "");
            type = vk::ImageType::e2D;
            break;
        case Type::Texture3D:
            VE_ASSERT(mSampleCount == 1, "");
            type = vk::ImageType::e3D;
            break;
        case Type::TextureCube:
            VE_ASSERT(mExtent3D.depth == 1 && mSampleCount == 1, "");
            type = vk::ImageType::e2D;
            break;
        case Type::Texture2DMultisample:
            VE_ASSERT(mExtent3D.depth == 1, "");
            type = vk::ImageType::e2D;
            break;
        default: throw ::std::runtime_error("ERROR: Invalid Texture Type.");
    }

    vk::ImageCreateInfo imageCreateInfo {};
    imageCreateInfo.setImageType(type)
        .setFormat(mFormat)
        .setExtent(mExtent3D)
        .setUsage(mUsageFlags)
        .setMipLevels(mMipLevels)
        .setArrayLayers(mArraySize);

    /* https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html */
    VmaAllocationCreateInfo allocCreateInfo {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VK_CHECK((vk::Result)vmaCreateImage(
        pAllocator->GetHandle(),
        reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
        &allocCreateInfo, &image, &mAllocation, &mAllocationInfo));

    return image;
}

}  // namespace IntelliDesign_NS::Vulkan::Core