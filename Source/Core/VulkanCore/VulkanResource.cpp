#include "VulkanResource.h"

#include "VulkanDevice.hpp"
#include "VulkanTexture.hpp"

VulkanResource::VulkanResource(VulkanDevice* device,
                               VulkanMemoryAllocator* allocator, Type type,
                               size_t size, vk::BufferUsageFlags usage,
                               BufferMemoryType memType)
    : mResource(std::in_place_type<VulkanBuffer>, device, allocator, size,
                usage, memType),
      mType(type) {}

VulkanResource::VulkanResource(VulkanDevice* device,
                               VulkanMemoryAllocator* allocator, Type type,
                               vk::Format format, vk::Extent3D extent,
                               vk::ImageUsageFlags usage, uint32_t mipLevels,
                               uint32_t arraySize, uint32_t sampleCount)
    : mResource(std::in_place_type<VulkanTexture>, device, allocator,
                static_cast<VulkanTexture::Type>(type), format, extent, usage,
                mipLevels, arraySize, sampleCount),
      mType(type) {}

VulkanResource::VulkanResource(VulkanDevice* device, vk::Image handle,
                               Type type, vk::Format format,
                               vk::Extent3D extent, uint32_t arraySize,
                               uint32_t sampleCount)
    : mResource(std::in_place_type<VulkanTexture>, device, handle,
                static_cast<VulkanTexture::Type>(type), format, extent,
                arraySize, sampleCount),
      mType(type) {}

VulkanResource::Type VulkanResource::GetType() const {
    return mType;
}

void VulkanResource::SetName(std::string const& name) {

    mName = name;
}

vk::Buffer VulkanResource::GetBufferHandle() const {
    return ::std::get<VulkanBuffer>(mResource).GetHandle();
}

void* VulkanResource::GetBufferMappedPtr() const {
    return ::std::get<VulkanBuffer>(mResource).GetMapPtr();
}

vk::DeviceAddress VulkanResource::GetBufferDeviceAddress() const {
    return ::std::get<VulkanBuffer>(mResource).GetDeviceAddress();
}

vk::BufferUsageFlags VulkanResource::GetBufferUsageFlags() const {
    return ::std::get<VulkanBuffer>(mResource).GetUsageFlags();
}

size_t VulkanResource::GetBufferSize() const {
    return ::std::get<VulkanBuffer>(mResource).GetSize();
}

BufferMemoryType VulkanResource::GetBufferMemType() const {
    return ::std::get<VulkanBuffer>(mResource).GetMemoryType();
}

void VulkanResource::CreateTextureView(std::string const& name,
                                       vk::ImageAspectFlags aspect,
                                       uint32_t mostDetailedMip,
                                       uint32_t mipCount, uint32_t firstArray,
                                       uint32_t arraySize) {
    ::std::get<VulkanTexture>(mResource).CreateImageView(
        name, aspect, mostDetailedMip, mipCount, firstArray, arraySize);
}

vk::Image VulkanResource::GetTextureHandle() const {
    return ::std::get<VulkanTexture>(mResource).GetHandle();
}

vk::ImageView VulkanResource::GetTextureViewHandle(
    std::string const& name) const {
    return ::std::get<VulkanTexture>(mResource).GetViewHandle(name);
}

uint32_t VulkanResource::GetWidth(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetWidth(mipLevel);
}

uint32_t VulkanResource::GetHeight(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetHeight(mipLevel);
}

uint32_t VulkanResource::GetDepth(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetDepth(mipLevel);
}

uint32_t VulkanResource::GetMipCount() const {
    return ::std::get<VulkanTexture>(mResource).GetMipCount();
}

uint32_t VulkanResource::GetSampleCount() const {
    return ::std::get<VulkanTexture>(mResource).GetSampleCount();
}

uint32_t VulkanResource::GetArraySize() const {
    return ::std::get<VulkanTexture>(mResource).GetArraySize();
}

vk::Format VulkanResource::GetFormat() const {
    return ::std::get<VulkanTexture>(mResource).GetFormat();
}

std::string_view VulkanResource::GetName() const {
    return mName;
}
