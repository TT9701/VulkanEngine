#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <cstdint>

#include "device_launch_parameters.h"

#include <vma/vk_mem_alloc.h>

namespace vk {
class Device;
}

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

    vk::Buffer GetBuffer() const { return mBuffer; }

    cudaExternalMemory_t GetCUDAExternalMemory() const {
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

void* MapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                  unsigned long long   offset,
                                  unsigned long long   size);

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

cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromFileDescriptor(int fd);

cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromNtHandle(HANDLE handle);

cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromNamedNtHandle(
    LPCWSTR name);

cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromKmtHandle(HANDLE handle);

void SignalExternalSemaphore(cudaExternalSemaphore_t extSem,
                             cudaStream_t            stream);

void WaitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream);

void SimPoint(void* data, float time);

}  // namespace CUDA
