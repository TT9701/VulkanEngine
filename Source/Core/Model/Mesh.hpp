#pragma once

#include <glm/glm.hpp>

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "MeshType.hpp"
#include "Core/VulkanCore/VulkanBuffer.hpp"

struct GPUMeshBuffers {
    SharedPtr<VulkanBuffer> mIndexBuffer {nullptr};
    SharedPtr<VulkanBuffer> mVertexBuffer {nullptr};
    vk::DeviceAddress       mVertexBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    SharedPtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    SharedPtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress                     mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4         mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};