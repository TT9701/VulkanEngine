#pragma once

#ifdef CUDA_VULKAN_INTEROP
#include "CUDA/CUDAVulkan.h"
#endif

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Buffer.h"

namespace IntelliDesign_NS::Vulkan::Core {

struct GPUMeshBuffers {
    SharedPtr<Buffer> mVPBuf {nullptr};
    SharedPtr<Buffer> mVNBuf {nullptr};
    SharedPtr<Buffer> mVTBuf {nullptr};
    SharedPtr<Buffer> mMeshletBuf {nullptr};
    SharedPtr<Buffer> mMeshletTriBuf {nullptr};
    SharedPtr<Buffer> mMeshDataBuf {nullptr};
    SharedPtr<Buffer> mBoundingBoxBuf {nullptr};

    SharedPtr<Buffer> mMaterialBuf {nullptr};
    SharedPtr<Buffer> mMeshMaterialIdxBuf {nullptr};

    vk::DeviceAddress mVPBufAddr {};
    vk::DeviceAddress mVNBufAddr {};
    vk::DeviceAddress mVTBufAddr {};
    vk::DeviceAddress mMeshletBufAddr {};
    vk::DeviceAddress mMeshletTriBufAddr {};
    vk::DeviceAddress mMeshDataBufAddr {};
    vk::DeviceAddress mBoundingBoxBufAddr {};

    vk::DeviceAddress mMaterialBufAddr {};
    vk::DeviceAddress mMeshMaterialIdxBufAddr {};
};

#ifdef CUDA_VULKAN_INTEROP
struct ExternalGPUMeshBuffers {
    SharedPtr<CUDA::VulkanExternalBuffer> mIndexBuffer {};
    SharedPtr<CUDA::VulkanExternalBuffer> mVertexBuffer {};
    vk::DeviceAddress mVertexBufferAddress {};
};
#endif

}  // namespace IntelliDesign_NS::Vulkan::Core