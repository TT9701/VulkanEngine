#include "VulkanCUDAInterop.h"

#include <cmath>

#include "../Core/MeshType.hpp"

namespace {

__global__ void SimpleAdd(void* data, float time) {
    static_cast<Vertex*>(data)[2].position = {0.0f, sin(time), 0.0f};
}

}


namespace CUDA {

int GetCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice) {
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice,
                                   &vkPhysicalDeviceProperties2);

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

cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(
    int fd, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t         extMem = NULL;
    cudaExternalMemoryHandleDesc desc   = {};

    memset(&desc, 0, sizeof(desc));

    desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size      = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it

    return extMem;
}

cudaExternalMemory_t ImportVulkanMemoryObjectFromNtHandle(
    HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t         extMem = NULL;
    cudaExternalMemoryHandleDesc desc   = {};

    memset(&desc, 0, sizeof(desc));

    desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size                = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extMem;
}

void* MapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                  unsigned long long   offset,
                                  unsigned long long   size) {

    void* ptr = NULL;

    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size   = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    return ptr;
}

void SimPoint(void* data, float time) {
    SimpleAdd<<<1, 1>>>(data, time);
}

}  // namespace CUDA
