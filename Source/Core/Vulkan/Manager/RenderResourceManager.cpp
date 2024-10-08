#include "RenderResourceManager.hpp"

#include "Core/Vulkan/Native/Device.hpp"
#include "DescriptorManager.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

RenderResourceManager::Fn_SizeRelation RenderResourceManager::sFullSize {
    [](vk::Extent2D extent) {
        return extent;
    }};

RenderResourceManager::RenderResourceManager(Device* device,
                                             MemoryAllocator* allocator,
                                             DescriptorManager* manager)
    : pDevice(device), pAllocator(allocator), pDescManager(manager) {}

RenderResource* RenderResourceManager::CreateBuffer(const char* name,
                                                    size_t size,
                                                    vk::BufferUsageFlags usage,
                                                    Buffer::MemoryType memType,
                                                    size_t texelSize) {
    auto ptr = MakeShared<RenderResource>(pDevice, pAllocator,
                                          RenderResource::Type::Buffer, size,
                                          usage, memType, texelSize);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::CreateTexture(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount) {
    auto ptr =
        MakeShared<RenderResource>(pDevice, pAllocator, type, format, extent,
                                   usage, mipLevels, arraySize, sampleCount);

    ptr->SetName(name);

    mResources.emplace(name, ptr);

    return ptr.get();
}

RenderResource* RenderResourceManager::CreateScreenSizeRelatedBuffer(
    const char* name, size_t size, vk::BufferUsageFlags usage,
    Buffer::MemoryType memType, size_t texelSize, Fn_SizeRelation fn) {
    auto ptr = CreateBuffer(name, size, usage, memType, texelSize);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ptr;
}

RenderResource* RenderResourceManager::CreateScreenSizeRelatedTexture(
    const char* name, RenderResource::Type type, vk::Format format,
    vk::Extent3D extent, vk::ImageUsageFlags usage, uint32_t mipLevels,
    uint32_t arraySize, uint32_t sampleCount, Fn_SizeRelation fn) {
    auto ptr = CreateTexture(name, type, format, extent, usage, mipLevels,
                             arraySize, sampleCount);
    mScreenSizeRalatedResources.emplace(name, fn);
    return ptr;
}

RenderResource* RenderResourceManager::operator[](const char* name) const {
    return mResources.at(name).get();
}

void RenderResourceManager::CreateDescriptorSet(
    const char* name, uint32_t setIndex, const char* pipelineName,
    vk::ShaderStageFlags stage, Type_STLVector<DescriptorSetData> const& datas,
    uint32_t bufferIndex) {
    Type_STLVector<DescriptorSetData> d {datas};
    std::ranges::sort(
        d, [](DescriptorSetData const& l, DescriptorSetData const& r) {
            return l.binding < r.binding;
        });

    Type_STLString descSetLayoutName;
    descSetLayoutName.append(pipelineName);
    auto s = vk::to_string(stage);
    std::erase(s, ' ');
    descSetLayoutName.append("@" + s);
    descSetLayoutName.append("@Set" + ::std::to_string(setIndex));

    // create descriptor set
    pDescManager->CreateDescriptorSet(name, descSetLayoutName.c_str(),
                                      bufferIndex);

    // allocate descriptors
    for (auto& data : d) {
        auto resource = mResources.at(data.resource).get();
        if (::std::holds_alternative<Buffer>(resource->GetResource())) {
            resource->AllocateBufferDescriptor(pDescManager, data.binding, name,
                                               data.type);
        } else if (::std::holds_alternative<Texture>(resource->GetResource())) {
            Type_STLString view {};
            if (data.imageView.has_value()) {
                view = data.imageView.value();
            } else {
                throw ::std::runtime_error("image resource must has a view");
            }
            Sampler* sampler = nullptr;
            if (data.sampler.has_value()) {
                sampler = data.sampler.value();
            }
            resource->AllocateImageDescriptor(pDescManager, data.binding, name,
                                              data.type, view.c_str(), sampler);
        }
    }
}

void RenderResourceManager::ResizeScreenSizeRelatedResources(
    vk::Extent2D extent) const {
    for (auto& screenSizeResource : mScreenSizeRalatedResources) {
        auto fn = screenSizeResource.second;
        auto newExtent = fn(extent);
        mResources.at(screenSizeResource.first)
            ->Resize(pDescManager, newExtent);
    }
}

RenderResourceManager::Type_Resource_SizeRelations const&
RenderResourceManager::GetSrcreenSizeRelatedResources() const {
    return mScreenSizeRalatedResources;
}

}  // namespace IntelliDesign_NS::Vulkan::Core