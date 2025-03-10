#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class DescriptorManager;
class DGCSequenceBase;

class Buffer {
public:
    enum class MemoryType { DeviceLocal, Staging, ReadBack };

    Buffer(VulkanContext& context, size_t size, vk::BufferUsageFlags usage,
           MemoryType memType, size_t stride = 1);

    ~Buffer();

public:
    vk::Buffer GetHandle() const;

    // for buffers include vk::BufferUsageFlagBits::eShaderDeviceAddress flagbits
    // else return 0
    vk::DeviceAddress GetDeviceAddress() const;

    vk::BufferUsageFlags GetUsageFlags() const;

    size_t GetSize() const;

    size_t GetStride() const;

    virtual uint32_t GetCount() const;

    MemoryType GetMemoryType() const;

    // for mapped buffers, non mapped buffers return nullptr
    void* GetMapPtr() const;

    void SetDGCSequence(SharedPtr<DGCSequenceBase> const& dgcSeq);

    void SetName(const char* name) const;

    void Resize(size_t newSize);

    void CopyData(const void* data, size_t size, size_t offset = 0);

    void Execute(vk::CommandBuffer cmd) const;

protected:
    vk::Buffer CreateBufferResource();

    void Destroy();

protected:
    VulkanContext& mContext;

    vk::BufferUsageFlags mUsageFlags;
    MemoryType mMemoryType;

    size_t mSize;
    size_t mStride;

    bool bMapped {false};
    bool bDeviceAddressEnabled {false};

    VmaAllocation mAllocation {};
    VmaAllocationInfo mAllocationInfo {};

    SharedPtr<DGCSequenceBase> mDGCSequence {};

    vk::Buffer mHandle;
};

class InstructionBuffer : public Buffer {};

template <class T>
class StructuredBuffer : public Buffer {
public:
    StructuredBuffer(VulkanContext& context, uint32_t count,
                     vk::BufferUsageFlags usage, MemoryType memType)
        : Buffer(context, sizeof(T) * count, usage, memType, sizeof(T)),
          mCount(count) {}

    uint32_t GetCount() const override { return mCount; }

    using Type = T;

private:
    uint32_t mCount;
};

}  // namespace IntelliDesign_NS::Vulkan::Core