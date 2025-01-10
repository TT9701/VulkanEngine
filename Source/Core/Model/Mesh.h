#pragma once

#include <meshoptimizer.h>
#include <glm/glm.hpp>

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAVulkan.h"
#endif

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/RenderResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

struct GPUMeshBuffers {
    SharedPtr<RenderResource> mVPBuf {nullptr};
    SharedPtr<RenderResource> mVNBuf {nullptr};
    SharedPtr<RenderResource> mVTBuf {nullptr};
    SharedPtr<RenderResource> mMeshletBuf {nullptr};
    SharedPtr<RenderResource> mMeshletVertBuf {nullptr};
    SharedPtr<RenderResource> mMeshletTriBuf {nullptr};
    SharedPtr<RenderResource> mMeshDataBuf {nullptr};
    SharedPtr<RenderResource> mBoundingBoxBuf {nullptr};

    vk::DeviceAddress mVPBufAddr {};
    vk::DeviceAddress mVNBufAddr {};
    vk::DeviceAddress mVTBufAddr {};
    vk::DeviceAddress mMeshletBufAddr {};
    vk::DeviceAddress mMeshletVertBufAddr {};
    vk::DeviceAddress mMeshletTriBufAddr {};
    vk::DeviceAddress mMeshDataBufAddr {};
    vk::DeviceAddress mBoundingBoxBufAddr {};
};

#ifdef CUDA_VULKAN_INTEROP
struct ExternalGPUMeshBuffers {
    SharedPtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    SharedPtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};
#endif

}  // namespace IntelliDesign_NS::Vulkan::Core