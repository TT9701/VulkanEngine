#include "VulkanResource.h"

#include "VulkanDevice.hpp"
#include "VulkanTexture.hpp"

VulkanRenderResource::VulkanRenderResource(VulkanDevice* device,
                               VulkanMemoryAllocator* allocator, Type type,
                               size_t size, vk::BufferUsageFlags usage,
                               VulkanBuffer::MemoryType memType)
    : mResource(std::in_place_type<VulkanBuffer>, device, allocator, size,
                usage, memType),
      mType(type) {}

VulkanRenderResource::VulkanRenderResource(VulkanDevice* device,
                               VulkanMemoryAllocator* allocator, Type type,
                               vk::Format format, vk::Extent3D extent,
                               vk::ImageUsageFlags usage, uint32_t mipLevels,
                               uint32_t arraySize, uint32_t sampleCount)
    : mResource(std::in_place_type<VulkanTexture>, device, allocator,
                static_cast<VulkanTexture::Type>(type), format, extent, usage,
                mipLevels, arraySize, sampleCount),
      mType(type) {}

VulkanRenderResource::VulkanRenderResource(VulkanDevice* device, vk::Image handle,
                               Type type, vk::Format format,
                               vk::Extent3D extent, uint32_t arraySize,
                               uint32_t sampleCount)
    : mResource(std::in_place_type<VulkanTexture>, device, handle,
                static_cast<VulkanTexture::Type>(type), format, extent,
                arraySize, sampleCount),
      mType(type) {}

VulkanRenderResource::Type VulkanRenderResource::GetType() const {
    return mType;
}

void VulkanRenderResource::SetName(const char* name) {

    mName = name;
}

vk::Buffer VulkanRenderResource::GetBufferHandle() const {
    return ::std::get<VulkanBuffer>(mResource).GetHandle();
}

void* VulkanRenderResource::GetBufferMappedPtr() const {
    return ::std::get<VulkanBuffer>(mResource).GetMapPtr();
}

vk::DeviceAddress VulkanRenderResource::GetBufferDeviceAddress() const {
    return ::std::get<VulkanBuffer>(mResource).GetDeviceAddress();
}

vk::BufferUsageFlags VulkanRenderResource::GetBufferUsageFlags() const {
    return ::std::get<VulkanBuffer>(mResource).GetUsageFlags();
}

size_t VulkanRenderResource::GetBufferSize() const {
    return ::std::get<VulkanBuffer>(mResource).GetSize();
}

VulkanBuffer::MemoryType VulkanRenderResource::GetBufferMemType() const {
    return ::std::get<VulkanBuffer>(mResource).GetMemoryType();
}

void VulkanRenderResource::CreateTexView(std::string_view name,
                                   vk::ImageAspectFlags aspect,
                                   uint32_t mostDetailedMip, uint32_t mipCount,
                                   uint32_t firstArray, uint32_t arraySize) {
    ::std::get<VulkanTexture>(mResource).CreateImageView(
        name, aspect, mostDetailedMip, mipCount, firstArray, arraySize);
}

vk::Image VulkanRenderResource::GetTexHandle() const {
    return ::std::get<VulkanTexture>(mResource).GetHandle();
}

vk::ImageView VulkanRenderResource::GetTexViewHandle(std::string_view name) const {
    return ::std::get<VulkanTexture>(mResource).GetViewHandle(name);
}

uint32_t VulkanRenderResource::GetTexWidth(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetWidth(mipLevel);
}

uint32_t VulkanRenderResource::GetTexHeight(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetHeight(mipLevel);
}

uint32_t VulkanRenderResource::GetTexDepth(uint32_t mipLevel) const {
    return ::std::get<VulkanTexture>(mResource).GetDepth(mipLevel);
}

uint32_t VulkanRenderResource::GetTexMipCount() const {
    return ::std::get<VulkanTexture>(mResource).GetMipCount();
}

uint32_t VulkanRenderResource::GetTexSampleCount() const {
    return ::std::get<VulkanTexture>(mResource).GetSampleCount();
}

uint32_t VulkanRenderResource::GetTexArraySize() const {
    return ::std::get<VulkanTexture>(mResource).GetArraySize();
}

vk::Format VulkanRenderResource::GetTexFormat() const {
    return ::std::get<VulkanTexture>(mResource).GetFormat();
}

std::string_view VulkanRenderResource::GetName() const {
    return mName;
}
