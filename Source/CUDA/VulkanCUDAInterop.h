#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "device_launch_parameters.h"

namespace CUDA {

int GetCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice);

cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(
    int fd, unsigned long long size, bool isDedicated);

cudaExternalMemory_t ImportVulkanMemoryObjectFromNtHandle(
    HANDLE handle, unsigned long long size, bool isDedicated);

void* MapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                  unsigned long long   offset,
                                  unsigned long long   size);

void SimPoint(void* data, float time);

}  // namespace CUDA
