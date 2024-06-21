#define VULKAN_HPP_NO_SPACESHIP_OPERATOR
#define VULKAN_HPP_NO_TO_STRING
#include "CUDAVulkan.h"

#include <vulkan/vulkan_win32.h>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../Core/MeshType.hpp"

namespace {

__global__ void SimpleAdd(void* data, float time) {
    auto vertices = static_cast<Vertex*>(data);

    vertices[0].position = {0.0f, 0.0f, 0.0f};
    vertices[1].position = {1.0f, 0.0f, 0.0f};
    vertices[2].position = {0.5f, 0.5f * ::std::sin(time / 1000.0f), 0.0f};

    vertices[0].color = {1.0f, 0.0f, 0.0f, 1.0f};
    vertices[1].color = {0.0f, 1.0f, 0.0f, 1.0f};
    vertices[2].color = {0.0f, 0.0f, 1.0f, 1.0f};

    vertices[0].uvX = 1.0f;
    vertices[1].uvX = 0.0f;
    vertices[2].uvX = 0.5f;

    vertices[0].uvY = 0.0f;
    vertices[1].uvY = 0.0f;
    vertices[2].uvY = 1.0f;
}

cudaChannelFormatDesc GetCudaChannelFormatDescForVulkanFormat(
    vk::Format format) {
    cudaChannelFormatDesc d;

    memset(&d, 0, sizeof(d));

    switch (format) {
        case vk::Format::eR8Uint:
            d.x = 8;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR8Sint:
            d.x = 8;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR8G8Uint:
            d.x = 8;
            d.y = 8;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR8G8Sint:
            d.x = 8;
            d.y = 8;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR8G8B8A8Uint:
            d.x = 8;
            d.y = 8;
            d.z = 8;
            d.w = 8;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR8G8B8A8Sint:
            d.x = 8;
            d.y = 8;
            d.z = 8;
            d.w = 8;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR16Uint:
            d.x = 16;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR16Sint:
            d.x = 16;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR16G16Uint:
            d.x = 16;
            d.y = 16;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR16G16Sint:
            d.x = 16;
            d.y = 16;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR16G16B16A16Uint:
            d.x = 16;
            d.y = 16;
            d.z = 16;
            d.w = 16;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR16G16B16A16Sint:
            d.x = 16;
            d.y = 16;
            d.z = 16;
            d.w = 16;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR16G16B16A16Sfloat:
            d.x = 16;
            d.y = 16;
            d.z = 16;
            d.w = 16;
            d.f = cudaChannelFormatKindFloat;
            break;
        case vk::Format::eR32Uint:
            d.x = 32;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR32Sint:
            d.x = 32;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR32Sfloat:
            d.x = 32;
            d.y = 0;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindFloat;
            break;
        case vk::Format::eR32G32Uint:
            d.x = 32;
            d.y = 32;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR32G32Sint:
            d.x = 32;
            d.y = 32;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR32G32Sfloat:
            d.x = 32;
            d.y = 32;
            d.z = 0;
            d.w = 0;
            d.f = cudaChannelFormatKindFloat;
            break;
        case vk::Format::eR32G32B32A32Uint:
            d.x = 32;
            d.y = 32;
            d.z = 32;
            d.w = 32;
            d.f = cudaChannelFormatKindUnsigned;
            break;
        case vk::Format::eR32G32B32A32Sint:
            d.x = 32;
            d.y = 32;
            d.z = 32;
            d.w = 32;
            d.f = cudaChannelFormatKindSigned;
            break;
        case vk::Format::eR32G32B32A32Sfloat:
            d.x = 32;
            d.y = 32;
            d.z = 32;
            d.w = 32;
            d.f = cudaChannelFormatKindFloat;
            break;
        default:
            assert(0);
    }
    return d;
}

cudaExtent GetCudaExtentForVulkanExtent(vk::Extent3D      vkExt,
                                        uint32_t          arrayLayers,
                                        vk::ImageViewType vkImageViewType) {
    cudaExtent e = {0, 0, 0};

    switch (vkImageViewType) {
        case vk::ImageViewType::e1D:
            e.width  = vkExt.width;
            e.height = 0;
            e.depth  = 0;
            break;
        case vk::ImageViewType::e2D:
            e.width  = vkExt.width;
            e.height = vkExt.height;
            e.depth  = 0;
            break;
        case vk::ImageViewType::e3D:
            e.width  = vkExt.width;
            e.height = vkExt.height;
            e.depth  = vkExt.depth;
            break;
        case vk::ImageViewType::eCube:
            e.width  = vkExt.width;
            e.height = vkExt.height;
            e.depth  = arrayLayers;
            break;
        case vk::ImageViewType::e1DArray:
            e.width  = vkExt.width;
            e.height = 0;
            e.depth  = arrayLayers;
            break;
        case vk::ImageViewType::e2DArray:
            e.width  = vkExt.width;
            e.height = vkExt.height;
            e.depth  = arrayLayers;
            break;
        case vk::ImageViewType::eCubeArray:
            e.width  = vkExt.width;
            e.height = vkExt.height;
            e.depth  = arrayLayers;
            break;
        default:
            assert(0);
    }

    return e;
}

unsigned int GetCudaMipmappedArrayFlagsForVulkanImage(
    vk::ImageViewType vkImageViewType, vk::ImageUsageFlags vkImageUsageFlags,
    bool allowSurfaceLoadStore) {
    unsigned int flags = 0;

    switch (vkImageViewType) {
        case vk::ImageViewType::eCube:
            flags |= cudaArrayCubemap;
            break;
        case vk::ImageViewType::eCubeArray:
            flags |= cudaArrayCubemap | cudaArrayLayered;
            break;
        case vk::ImageViewType::e1DArray:
            flags |= cudaArrayLayered;
            break;
        case vk::ImageViewType::e2DArray:
            flags |= cudaArrayLayered;
            break;
        default:
            break;
    }

    if (vkImageUsageFlags & vk::ImageUsageFlagBits::eColorAttachment) {
        flags |= cudaArrayColorAttachment;
    }

    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }
    return flags;
}

cudaExternalMemory_t GetCUDAExternalMemory(VkDevice       device,
                                           VkDeviceMemory vkDeviceMemory,
                                           size_t         allocByteSize) {
    PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
            device, "vkGetMemoryWin32HandleKHR");

    if (!fpGetMemoryWin32HandleKHR) {
        throw std::runtime_error(
            "Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }

    VkMemoryGetWin32HandleInfoKHR memoryWin32HandleInfo {};
    memoryWin32HandleInfo.sType =
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    memoryWin32HandleInfo.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    memoryWin32HandleInfo.memory = vkDeviceMemory;

    HANDLE handle {};

    if (fpGetMemoryWin32HandleKHR(device, &memoryWin32HandleInfo, &handle) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    cudaExternalMemoryHandleDesc desc {};

    memset(&desc, 0, sizeof(desc));

    desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size                = allocByteSize;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaExternalMemory_t extMem {};
    cudaImportExternalMemory(&extMem, &desc);

    CloseHandle(handle);

    return extMem;
}

}  // namespace

namespace CUDA {

int GetVulkanCUDABindDeviceID(vk::PhysicalDevice vkPhysicalDevice) {
    vk::PhysicalDeviceIDProperties vkPhysicalDeviceIDProperties {};

    vk::PhysicalDeviceProperties2 vkPhysicalDeviceProperties2 {};
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;
    vkPhysicalDevice.getProperties2(&vkPhysicalDeviceProperties2);

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID,
                    VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

void VulkanExternalBuffer::CreateExternalBuffer(
    vk::Device device, VmaAllocator allocator, size_t allocByteSize,
    vk::BufferUsageFlags usage, VmaAllocationCreateFlags flags, VmaPool pool) {
    vk::ExternalMemoryBufferCreateInfo externalbuffer {};
    externalbuffer.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);

    vk::BufferCreateInfo bufferInfo {};
    bufferInfo.setSize(allocByteSize).setUsage(usage).setPNext(&externalbuffer);

    VmaAllocationCreateInfo vmaAllocInfo {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaAllocInfo.flags = flags;
    vmaAllocInfo.pool  = pool;

    vmaCreateBuffer(allocator, (VkBufferCreateInfo*)&bufferInfo, &vmaAllocInfo,
                    (VkBuffer*)&mBuffer, &mAllocation, &mInfo);

    mExternalMemory =
        GetCUDAExternalMemory(device, mInfo.deviceMemory, mInfo.size);

    mAllocator = allocator;
}

VulkanMappedPointer VulkanExternalBuffer::GetMappedPointer(size_t offset,
                                                           size_t size) const {
    cudaExternalMemoryBufferDesc desc {offset, size, 0};
    void*                        ptr = nullptr;
    cudaExternalMemoryGetMappedBuffer(&ptr, mExternalMemory, &desc);
    return {ptr};
}

void VulkanExternalBuffer::Destroy() {
    vmaDestroyBuffer(mAllocator, mBuffer, mAllocation);
}

void VulkanExternalImage::CreateExternalImage(
    vk::Device device, VmaAllocator allocator, VmaPool pool,
    VmaAllocationCreateFlags flags, vk::Extent3D extent, vk::Format format,
    vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect, bool mipmaped,
    uint32_t arrayLayers, vk::ImageType type, vk::ImageViewType viewType) {
    mExtent3D = extent;
    mFormat   = format;

    uint32_t mipLevels =
        mipmaped ? static_cast<uint32_t>(1 + ::std::floor(::std::log2(std::max(
                                                 extent.width, extent.height))))
                 : 1;

    vk::ExternalMemoryImageCreateInfo externalImage {};
    externalImage.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);

    vk::ImageCreateInfo imageCreateInfo {};
    imageCreateInfo.setImageType(type)
        .setFormat(mFormat)
        .setExtent(mExtent3D)
        .setUsage(usage)
        .setMipLevels(mipLevels)
        .setArrayLayers(arrayLayers)
        .setPNext(&externalImage);

    VmaAllocationCreateInfo allocCreateInfo {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = flags;
    allocCreateInfo.pool  = pool;

    vmaCreateImage(allocator,
                   reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo),
                   &allocCreateInfo, reinterpret_cast<VkImage*>(&mImage),
                   &mAllocation, &mInfo);

    vk::ImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.setViewType(viewType)
        .setImage(mImage)
        .setFormat(mFormat)
        .setSubresourceRange(
            {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});

    mImageView = device.createImageView(imageViewCreateInfo);

    mExternalMemory =
        GetCUDAExternalMemory(device, mInfo.deviceMemory, mInfo.size);

    memset(&mArrayDesc, 0, sizeof(mArrayDesc));
    auto formatDesc = GetCudaChannelFormatDescForVulkanFormat(mFormat);
    auto ext        = GetCudaExtentForVulkanExtent(
        mExtent3D, imageViewCreateInfo.subresourceRange.layerCount, viewType);
    auto descFlags = GetCudaMipmappedArrayFlagsForVulkanImage(
        viewType, usage,
        usage & vk::ImageUsageFlagBits::eStorage ? true : false);

    mArrayDesc.formatDesc = formatDesc;
    mArrayDesc.extent     = ext;
    mArrayDesc.flags      = descFlags;

    auto        mipmappedArray = GetMapMipmappedArray(0, 1);
    cudaArray_t cudaMipLevelArray {};
    cudaGetMipmappedArrayLevel(&cudaMipLevelArray, mipmappedArray, 0);

    cudaResourceDesc surfaceResDesc {};
    surfaceResDesc.resType         = cudaResourceTypeArray;
    surfaceResDesc.res.array.array = cudaMipLevelArray;

    mSurface2D =
        ::std::make_shared<CUDASurface2D<UserType, 1600, 900>>(&surfaceResDesc);
}

cudaMipmappedArray_t VulkanExternalImage::GetMapMipmappedArray(
    unsigned long long offset, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap {nullptr};

    mArrayDesc.offset    = offset;
    mArrayDesc.numLevels = numLevels;

    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, mExternalMemory,
                                              &mArrayDesc);

    return mipmap;
}

void VulkanExternalImage::Destroy(vk::Device device, VmaAllocator allocator) {
    vmaDestroyImage(allocator, mImage, mAllocation);
    device.destroy(mImageView);
}

void VulkanExternalSemaphore::CreateExternalSemaphore(vk::Device device) {
    vk::SemaphoreCreateInfo semaphoreInfo {};

    vk::ExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo {};
    exportSemaphoreCreateInfo.setHandleTypes(
        vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32);
    semaphoreInfo.setPNext(&exportSemaphoreCreateInfo);

    mSemaphore = device.createSemaphore(semaphoreInfo);

    VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR {};
    semaphoreGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    semaphoreGetWin32HandleInfoKHR.pNext     = nullptr;
    semaphoreGetWin32HandleInfoKHR.semaphore = mSemaphore;
    semaphoreGetWin32HandleInfoKHR.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
    fpGetSemaphoreWin32HandleKHR =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
            device, "vkGetSemaphoreWin32HandleKHR");

    if (!fpGetSemaphoreWin32HandleKHR) {
        throw std::runtime_error(
            "Failed to retrieve vkGetSemaphoreWin32HandleKHR!");
    }

    HANDLE handle;

    if (fpGetSemaphoreWin32HandleKHR(device, &semaphoreGetWin32HandleInfoKHR,
                                     &handle) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for Semaphore!");
    }

    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));

    desc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&mExternalSemaphore, &desc);

    CloseHandle(handle);
}

void VulkanExternalSemaphore::InsertWaitToStreamAsync(cudaStream_t cudaStream) {
    cudaExternalSemaphoreWaitParams params = {};
    memset(&params, 0, sizeof(params));

    cudaWaitExternalSemaphoresAsync(&mExternalSemaphore, &params, 1,
                                    cudaStream);
}

void VulkanExternalSemaphore::InsertSignalToStreamAsync(
    cudaStream_t cudaStream) {
    cudaExternalSemaphoreSignalParams params = {};
    memset(&params, 0, sizeof(params));

    cudaSignalExternalSemaphoresAsync(&mExternalSemaphore, &params, 1,
                                      cudaStream);
}

void VulkanExternalSemaphore::Destroy(vk::Device device) {
    device.destroy(mSemaphore);
}

cudaMipmappedArray_t MapMipmappedArrayOntoExternalMemory(
    cudaExternalMemory_t extMem, unsigned long long offset,
    cudaChannelFormatDesc* formatDesc, cudaExtent* extent, unsigned int flags,
    unsigned int numLevels) {
    cudaMipmappedArray_t                 mipmap {nullptr};
    cudaExternalMemoryMipmappedArrayDesc desc {};

    memset(&desc, 0, sizeof(desc));

    desc.offset     = offset;
    desc.formatDesc = *formatDesc;
    desc.extent     = *extent;
    desc.flags      = flags;
    desc.numLevels  = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}

void SimPoint(void* data, float time, cudaStream_t stream) {
    SimpleAdd<<<1, 1, 0, stream>>>(data, time);
}

void SimSurface(CUDASurface2D<UserType, 1600, 900> surf, float time,
                cudaStream_t stream) {
    SimSurfaceKernel<<<{100, 100}, {16, 16}, 0, stream>>>(surf, time);
}

}  // namespace CUDA
