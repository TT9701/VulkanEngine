#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <cstdint>

#include "device_launch_parameters.h"

#include <vma/vk_mem_alloc.h>

namespace CUDA {

int GetVulkanCUDABindDeviceID(vk::PhysicalDevice vkPhysicalDevice);

class VulkanMappedPointer {
public:
    VulkanMappedPointer(void* mapped) : mMapped(mapped) {}

    void* GetPtr() const { return mMapped; }

    ~VulkanMappedPointer() {
        if (mMapped)
            cudaFree(mMapped);
    }

private:
    void* mMapped {nullptr};
};

class VulkanExternalBuffer {
public:
    void CreateExternalBuffer(vk::Device device, VmaAllocator allocator,
                              size_t allocByteSize, vk::BufferUsageFlags usage,
                              VmaAllocationCreateFlags flags, VmaPool pool);

    VmaAllocationInfo const& GetAllocationInfo() const { return mInfo; }

    vk::Buffer const& GetVkBuffer() const { return mBuffer; }

    cudaExternalMemory_t GetExternalMemory() const {
        return mExternalMemory;
    }

    VulkanMappedPointer GetMappedPointer(size_t offset, size_t size) const {
        cudaExternalMemoryBufferDesc desc {offset, size, 0};
        void*                        ptr = nullptr;
        cudaExternalMemoryGetMappedBuffer(&ptr, mExternalMemory, &desc);
        return {ptr};
    }

    void Destroy() { vmaDestroyBuffer(mAllocator, mBuffer, mAllocation); }

private:
    VmaAllocator         mAllocator {};
    vk::Buffer           mBuffer {};
    VmaAllocation        mAllocation {};
    VmaAllocationInfo    mInfo {};
    cudaExternalMemory_t mExternalMemory {};
};

class VulkanExternalImage {
public:

private:

};

class VulkanExternalSemaphore {
public:
    void CreateExternalSemaphore(vk::Device device);

    void InsertWaitToStreamAsync(cudaStream_t cudaStream);

    void InsertSignalToStreamAsync(cudaStream_t cudaStream);

    vk::Semaphore const& GetVkSemaphore() const { return mSemaphore; }

    cudaExternalSemaphore_t const& GetCUDAExternalSemaphore() const {
        return mExternalSemaphore;
    }

    void Destroy(vk::Device device) { device.destroy(mSemaphore); }

private:
    cudaExternalSemaphore_t mExternalSemaphore {};
    vk::Semaphore           mSemaphore {};
};

cudaMipmappedArray_t MapMipmappedArrayOntoExternalMemory(
    cudaExternalMemory_t extMem, unsigned long long offset,
    cudaChannelFormatDesc* formatDesc, cudaExtent* extent, unsigned int flags,
    unsigned int numLevels);

cudaChannelFormatDesc GetCudaChannelFormatDescForVulkanFormat(
    vk::Format format);

cudaExtent GetCudaExtentForVulkanExtent(vk::Extent3D      vkExt,
                                        uint32_t          arrayLayers,
                                        vk::ImageViewType vkImageViewType);

unsigned int GetCudaMipmappedArrayFlagsForVulkanImage(
    vk::ImageViewType vkImageViewType, vk::ImageUsageFlags vkImageUsageFlags,
    bool allowSurfaceLoadStore);

void SimPoint(void* data, float time);

}  // namespace CUDA
