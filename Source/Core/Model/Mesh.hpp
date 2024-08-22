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
    vk::DeviceAddress mVertexBufferAddress {};
    vk::DeviceAddress mIndexBufferAddress {};
    vk::DeviceAddress mMeshletBufferAddress {};
    vk::DeviceAddress mMeshletVertBufferAddress {};
    vk::DeviceAddress mMeshletTriBufferAddress {};
};

struct ExternalGPUMeshBuffers {
    SharedPtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    SharedPtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};

class Mesh {
public:
    Mesh(::std::vector<Vertex> const& vertices,
         ::std::vector<uint32_t> const& indices);

    ::std::vector<Vertex> mVertices {};
    ::std::vector<uint32_t> mIndices {};

    ::std::vector<meshopt_Meshlet> mMeshlets {};
    ::std::vector<uint32_t> mMeshletVertices {};
    ::std::vector<uint8_t> mMeshletTriangles {};

    // TODO: Textures
};

}  // namespace IntelliDesign_NS::Vulkan::Core