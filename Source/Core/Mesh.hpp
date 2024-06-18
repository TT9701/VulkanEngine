#pragma once

#include "MeshType.hpp"
#include "VulkanBuffer.hpp"

struct GPUMeshBuffers {
    AllocatedVulkanBuffer mIndexBuffer {};
    AllocatedVulkanBuffer mVertexBuffer {};
    vk::DeviceAddress     mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4         mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};