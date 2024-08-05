#pragma once

#include <glm/glm.hpp>

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/VulkanCore/VulkanBuffer.hpp"
#include "MeshType.hpp"

class VulkanContext;
class VulkanEngine;

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
    glm::mat4         mModelMatrix {glm::mat4(1.0f)};
    vk::DeviceAddress mVertexBufferAddress {};
};

class Mesh {
public:
    Mesh(::std::vector<Vertex> const&   vertices,
         ::std::vector<uint32_t> const& indices);

    void GenerateBuffers(VulkanContext* context, VulkanEngine* engine);

    ::std::vector<Vertex> const& GetVertexData() const { return mVertices; }

    ::std::vector<uint32_t> const& GetIndexData() const { return mIndices; }

    VulkanBuffer* GetVertexBuffer() const {
        return mBuffers.mVertexBuffer.get();
    }

    VulkanBuffer* GetIndexBuffer() const { return mBuffers.mIndexBuffer.get(); }

    MeshPushConstants GetPushContants() const { return mConstants; }

private:
    ::std::vector<Vertex>   mVertices {};
    ::std::vector<uint32_t> mIndices {};
    // TODO: Textures
    GPUMeshBuffers    mBuffers {};
    MeshPushConstants mConstants {};
};