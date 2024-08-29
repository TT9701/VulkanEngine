#pragma once

#include <glm/glm.hpp>
#include <meshoptimizer.h>

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/RenderResource.hpp"
#include "MeshType.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

struct GPUMeshBuffers {
    SharedPtr<RenderResource> mIndexBuffer {nullptr};
    SharedPtr<RenderResource> mVertexBuffer {nullptr};
    SharedPtr<RenderResource> mMeshletBuffer {nullptr};
    SharedPtr<RenderResource> mMeshletVertBuffer {nullptr};
    SharedPtr<RenderResource> mMeshletTriBuffer {nullptr};
    SharedPtr<RenderResource> mMeshDataBuffer {nullptr};
    vk::DeviceAddress mVertexBufferAddress {};
    vk::DeviceAddress mMeshletBufferAddress {};
    vk::DeviceAddress mMeshletVertBufferAddress {};
    vk::DeviceAddress mMeshletTriBufferAddress {};
    vk::DeviceAddress mMeshDataBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    SharedPtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    SharedPtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};

class Mesh {
public:
    Mesh(Type_STLVector<Vertex> const& vertices,
         Type_STLVector<uint32_t> const& indices);

    Type_STLVector<Vertex> mVertices {};
    Type_STLVector<uint32_t> mIndices {};

    // meshlet datas
    Type_STLVector<meshopt_Meshlet> mMeshlets {};
    Type_STLVector<uint32_t> mMeshletVertices {};
    Type_STLVector<uint8_t> mMeshletTriangles {};

    // TODO: Textures
};

}  // namespace IntelliDesign_NS::Vulkan::Core