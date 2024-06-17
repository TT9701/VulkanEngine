#pragma once

#include <glm/glm.hpp>

#include "VulkanBuffer.hpp"

struct Vertex {
    glm::vec3 position {};
    float     uvX {};
    glm::vec3 normal {};
    float     uvY {};
    glm::vec4 color {};
};

struct GPUMeshBuffers {
    AllocatedVulkanBuffer mIndexBuffer {};
    AllocatedVulkanBuffer mVertexBuffer {};
    vk::DeviceAddress     mVertexBufferAddress {};
};

struct MeshPushConstants {
    glm::mat4         mModelMatrix;
    vk::DeviceAddress mVertexBufferAddress {};
};