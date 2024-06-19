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

    cudaExternalMemory_t GetExternalMemory() const { return mExternalMemory; }

    VulkanMappedPointer GetMappedPointer(size_t offset, size_t size) const;

    void Destroy();

private:
    VmaAllocator         mAllocator {};
    vk::Buffer           mBuffer {};
    VmaAllocation        mAllocation {};
    VmaAllocationInfo    mInfo {};
    cudaExternalMemory_t mExternalMemory {};
};

class VulkanExternalImage {
public:
    void CreateExternalImage(
        vk::Device device, VmaAllocator allocator, VmaPool pool,
        VmaAllocationCreateFlags flags, vk::Extent3D extent, vk::Format format,
        vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
        bool mipmaped = false, uint32_t arrayLayers = 1,
        vk::ImageType     type     = vk::ImageType::e2D,
        vk::ImageViewType viewType = vk::ImageViewType::e2D);

    cudaMipmappedArray_t GetMapMipmappedArray(unsigned long long offset,
                                              unsigned int       numLevels);

    vk::Image const& GetVkImage() const { return mImage; }

    vk::ImageView const& GetVkImageView() const { return mImageView; }

    VmaAllocationInfo const& GetVmaAllocationInfo() const { return mInfo; }

    vk::Extent3D const& GetExtent3D() const { return mExtent3D; }

    vk::Format const& GetFormat() const { return mFormat; }

    void Destroy(vk::Device device, VmaAllocator allocator);

private:
    vk::Image            mImage {};
    vk::ImageView        mImageView {};
    VmaAllocation        mAllocation {};
    VmaAllocationInfo    mInfo {};
    vk::Extent3D         mExtent3D {};
    vk::Format           mFormat {};
    vk::ImageLayout      mLayout {vk::ImageLayout::eUndefined};
    cudaExternalMemory_t mExternalMemory {};

    cudaExternalMemoryMipmappedArrayDesc mArrayDesc {};
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

    void Destroy(vk::Device device);

private:
    cudaExternalSemaphore_t mExternalSemaphore {};
    vk::Semaphore           mSemaphore {};
};

void SimPoint(void* data, float time);

}  // namespace CUDA
