#pragma once

#include <glm/glm.hpp>

#include "CUDA/CUDAVulkan.h"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/RenderResource.h"
#include "MeshType.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

struct GPUMeshBuffers {
    SharedPtr<RenderResource> mIndexBuffer {nullptr};
    SharedPtr<RenderResource> mVertexBuffer {nullptr};
    vk::DeviceAddress mVertexBufferAddress {};
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

    // TODO: Textures
};

}  // namespace IntelliDesign_NS::Vulkan::Core