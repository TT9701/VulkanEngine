#pragma once

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "MeshType.hpp"
#include "VulkanBuffer.hpp"

struct GPUMeshBuffers {
    USING_TEMPLATE_UNIQUE_PTR_TYPE(Type_PInstance);

    Type_PInstance<VulkanAllocatedBuffer> mIndexBuffer {nullptr};
    Type_PInstance<VulkanAllocatedBuffer> mVertexBuffer {nullptr};
    vk::DeviceAddress mVertexBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    CUDA::VulkanExternalBuffer mIndexBuffer {};
    CUDA::VulkanExternalBuffer mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4 mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};