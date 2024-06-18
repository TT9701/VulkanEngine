#pragma once

#include "MeshType.hpp"
#include "VulkanBuffer.hpp"
#include "CUDA/CUDAVulkan.h"

struct GPUMeshBuffers {
    AllocatedVulkanBuffer mIndexBuffer {};
    AllocatedVulkanBuffer mVertexBuffer {};
    vk::DeviceAddress     mVertexBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    CUDA::VulkanExternalBuffer mIndexBuffer {};
    CUDA::VulkanExternalBuffer mVertexBuffer {};
    vk::DeviceAddress          mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4         mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};