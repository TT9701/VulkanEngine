#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#if _WIN32
#include <Windows.h>
#endif

#include <cstdint>
#include <memory>

#include "device_launch_parameters.h"

#include <vma/vk_mem_alloc.h>

#include "CUDASurface.h"

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
    VulkanExternalBuffer(vk::Device device, VmaAllocator allocator,
                         size_t allocByteSize, vk::BufferUsageFlags usage,
                         VmaAllocationCreateFlags flags, VmaPool pool);
    ~VulkanExternalBuffer();

    VmaAllocationInfo GetAllocationInfo() const { return mInfo; }

    vk::Buffer GetVkBuffer() const { return mBuffer; }

    cudaExternalMemory_t GetExternalMemory() const { return mExternalMemory; }

    VulkanMappedPointer GetMappedPointer(size_t offset, size_t size) const;

private:
    VmaAllocator mAllocator;

    vk::Buffer           mBuffer {};
    VmaAllocation        mAllocation {};
    VmaAllocationInfo    mInfo {};
    cudaExternalMemory_t mExternalMemory {};
};

struct UserType {
    float a, b, c, d;
};

class VulkanExternalImage {
public:
    VulkanExternalImage(vk::Device device, VmaAllocator allocator, VmaPool pool,
                        VmaAllocationCreateFlags flags, vk::Extent3D extent,
                        vk::Format format, vk::ImageUsageFlags usage,
                        vk::ImageAspectFlags aspect, uint32_t mipmapLevels = 1,
                        uint32_t          arrayLayers = 1,
                        vk::ImageType     type        = vk::ImageType::e2D,
                        vk::ImageViewType viewType    = vk::ImageViewType::e2D);
    ~VulkanExternalImage();

    cudaMipmappedArray_t GetMapMipmappedArray(unsigned long long offset,
                                              unsigned int       numLevels);

public:
    vk::Image GetVkImage() const { return mImage; }

    vk::ImageView GetVkImageView() const { return mImageView; }

    VmaAllocationInfo GetVmaAllocationInfo() const { return mInfo; }

    vk::Extent3D GetExtent3D() const { return mExtent3D; }

    vk::Format GetFormat() const { return mFormat; }

    auto GetSurfaceObjectPtr() { return mSurface2D; }

private:
    vk::Device   mDevice;
    VmaAllocator mAllocator;

    vk::Image            mImage {};
    vk::ImageView        mImageView {};
    VmaAllocation        mAllocation {};
    VmaAllocationInfo    mInfo {};
    vk::Extent3D         mExtent3D {};
    vk::Format           mFormat {};
    vk::ImageLayout      mLayout {vk::ImageLayout::eUndefined};
    cudaExternalMemory_t mExternalMemory {};

    cudaExternalMemoryMipmappedArrayDesc                  mArrayDesc {};
    ::std::shared_ptr<CUDASurface2D<UserType, 1600, 900>> mSurface2D {};
};

class VulkanExternalSemaphore {
public:
    VulkanExternalSemaphore(vk::Device device);
    ~VulkanExternalSemaphore();

    void InsertWaitToStreamAsync(cudaStream_t cudaStream);

    void InsertSignalToStreamAsync(cudaStream_t cudaStream);

    vk::Semaphore GetVkSemaphore() const { return mSemaphore; }

    cudaExternalSemaphore_t GetCUDAExternalSemaphore() const {
        return mExternalSemaphore;
    }

private:
    vk::Device              pDevice;
    cudaExternalSemaphore_t mExternalSemaphore {};
    vk::Semaphore           mSemaphore {};
};

void SimPoint(void* data, float time, cudaStream_t stream);

template <typename UserType, int TexelWidth, int Height>
__global__ void SimSurfaceKernel(
    CUDASurface2D<UserType, TexelWidth, Height> surf, float time) {
    int x                        = threadIdx.x + blockIdx.x * blockDim.x;
    int y                        = threadIdx.y + blockIdx.y * blockDim.y;
    using TexelType              = CudaTexelTypeBinder_t<UserType>;
    constexpr int texelElemCount = sizeof(TexelType) / sizeof(float);

    float color = 0.5f
                + 0.5f
                      * ::std::sin((float)x / (blockDim.x * gridDim.x)
                                   - (float)y / (blockDim.y * gridDim.y)
                                   + time / 300.0f);

    CudaSurfaceArray2D<float, texelElemCount, 1> write {};
    write.data[0].data[0] = color;
    write.data[0].data[1] = color;
    write.data[0].data[2] = color;
    write.data[0].data[3] = 1.0f;
    surf.Write2D<0, texelElemCount, 0, 1>(write, x, y);
}

void SimSurface(CUDASurface2D<UserType, 1600, 900> surf, float time,
                cudaStream_t stream);

}  // namespace CUDA
