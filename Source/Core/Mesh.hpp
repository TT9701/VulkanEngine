#pragma once

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "MeshType.hpp"
#include "VulkanBuffer.hpp"

struct GPUMeshBuffers {
    UniquePtr<VulkanAllocatedBuffer> mIndexBuffer {nullptr};
    UniquePtr<VulkanAllocatedBuffer> mVertexBuffer {nullptr};
    vk::DeviceAddress mVertexBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    UniquePtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    UniquePtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4 mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};