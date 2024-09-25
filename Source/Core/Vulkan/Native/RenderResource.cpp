#include "RenderResource.hpp"

#include "Device.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResource::RenderResource(Device* device, MemoryAllocator* allocator,
                               Type type, size_t size,
                               vk::BufferUsageFlags usage,
                               Buffer::MemoryType memType, size_t texelSize)
    : mResource(std::in_place_type<Buffer>, device, allocator, size, usage,
                memType, texelSize),
      mType(type) {}

RenderResource::RenderResource(Device* device, MemoryAllocator* allocator,
                               Type type, vk::Format format,
                               vk::Extent3D extent, vk::ImageUsageFlags usage,
                               uint32_t mipLevels, uint32_t arraySize,
                               uint32_t sampleCount)
    : mResource(std::in_place_type<Texture>, device, allocator,
                static_cast<Texture::Type>(type), format, extent, usage,
                mipLevels, arraySize, sampleCount),
      mType(type) {}

RenderResource::RenderResource(Device* device, vk::Image handle, Type type,
                               vk::Format format, vk::Extent3D extent,
                               uint32_t arraySize, uint32_t sampleCount)
    : mResource(std::in_place_type<Texture>, device, handle,
                static_cast<Texture::Type>(type), format, extent, arraySize,
                sampleCount),
      mType(type) {}

RenderResource::RenderResource(Buffer&& buffer) : mResource(std::move(buffer)) {
    mType = Type::Buffer;
}

RenderResource::RenderResource(Texture&& texture)
    : mResource(std::move(texture)) {
    mType = static_cast<Type>(texture.GetType());
}

RenderResource::Type RenderResource::GetType() const {
    return mType;
}

template <class... Ts>
struct overload : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

void RenderResource::SetName(const char* name) {
    ::std::visit(overload {[&](Texture const& v) { v.SetName(name); },
                           [&](Buffer const& v) {
                               v.SetName(name);
                           }},
                 mResource);
    mName = name;
}

std::variant<Buffer, Texture> const& RenderResource::GetResource() const {
    return mResource;
}

vk::Buffer RenderResource::GetBufferHandle() const {
    return ::std::get<Buffer>(mResource).GetHandle();
}

void* RenderResource::GetBufferMappedPtr() const {
    return ::std::get<Buffer>(mResource).GetMapPtr();
}

vk::DeviceAddress RenderResource::GetBufferDeviceAddress() const {
    return ::std::get<Buffer>(mResource).GetDeviceAddress();
}

vk::BufferUsageFlags RenderResource::GetBufferUsageFlags() const {
    return ::std::get<Buffer>(mResource).GetUsageFlags();
}

size_t RenderResource::GetBufferSize() const {
    return ::std::get<Buffer>(mResource).GetSize();
}

Buffer::MemoryType RenderResource::GetBufferMemType() const {
    return ::std::get<Buffer>(mResource).GetMemoryType();
}

void RenderResource::CreateTexView(const char* name,
                                   vk::ImageAspectFlags aspect,
                                   uint32_t mostDetailedMip, uint32_t mipCount,
                                   uint32_t firstArray, uint32_t arraySize) {
    ::std::get<Texture>(mResource).CreateImageView(
        name, aspect, mostDetailedMip, mipCount, firstArray, arraySize);
}

vk::Image RenderResource::GetTexHandle() const {
    return ::std::get<Texture>(mResource).GetHandle();
}

vk::ImageView RenderResource::GetTexViewHandle(const char* name) const {
    return ::std::get<Texture>(mResource).GetViewHandle(name);
}

uint32_t RenderResource::GetTexWidth(uint32_t mipLevel) const {
    return ::std::get<Texture>(mResource).GetWidth(mipLevel);
}

uint32_t RenderResource::GetTexHeight(uint32_t mipLevel) const {
    return ::std::get<Texture>(mResource).GetHeight(mipLevel);
}

uint32_t RenderResource::GetTexDepth(uint32_t mipLevel) const {
    return ::std::get<Texture>(mResource).GetDepth(mipLevel);
}

uint32_t RenderResource::GetTexMipCount() const {
    return ::std::get<Texture>(mResource).GetMipCount();
}

uint32_t RenderResource::GetTexSampleCount() const {
    return ::std::get<Texture>(mResource).GetSampleCount();
}

uint32_t RenderResource::GetTexArraySize() const {
    return ::std::get<Texture>(mResource).GetArraySize();
}

vk::Format RenderResource::GetTexFormat() const {
    return ::std::get<Texture>(mResource).GetFormat();
}

void RenderResource::AllocateBufferDescriptor(DescriptorManager* manager,
                                              uint32_t binding,
                                              const char* descSetName,
                                              vk::DescriptorType type) const {
    ::std::get<Buffer>(mResource).AllocateDescriptor(manager, binding,
                                                     descSetName, type);
}

void RenderResource::AllocateImageDescriptor(
    DescriptorManager* manager, uint32_t binding, const char* descSetName,
    vk::DescriptorType type, const char* viewName, Sampler* sampler) {
    ::std::get<Texture>(mResource).AllocateDescriptor(
        manager, binding, descSetName, type, viewName, sampler);
}

void RenderResource::Resize(uint32_t width, uint32_t height) {
    ::std::visit(overload {[&](Texture& v) {
                               v.Resize(width, height);
                               v.SetName(mName.c_str());
                           },
                           [&](Buffer& v) {
                               v.Resize(width * height * v.GetTexelSize());
                               v.SetName(mName.c_str());
                           }},
                 mResource);
}

std::string_view RenderResource::GetName() const {
    return mName;
}

}  // namespace IntelliDesign_NS::Vulkan::Core