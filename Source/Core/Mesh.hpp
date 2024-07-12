#pragma once

#include "CUDA/CUDAVulkan.h"
#include "Core/System/MemoryPool/MemoryPool.h"
#include "MeshType.hpp"
#include "VulkanBuffer.hpp"

struct GPUMeshBuffers {
    IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<VulkanAllocatedBuffer>
        mIndexBuffer{nullptr};
    IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<VulkanAllocatedBuffer>
        mVertexBuffer{nullptr};
    vk::DeviceAddress mVertexBufferAddress{};
};

struct ExternalGPUMeshBuffers {
    CUDA::VulkanExternalBuffer mIndexBuffer{};
    CUDA::VulkanExternalBuffer mVertexBuffer{};
    vk::DeviceAddress mVertexBufferAddress{};
};

struct MeshPushConstants {
    glm::mat4 mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress{};
};