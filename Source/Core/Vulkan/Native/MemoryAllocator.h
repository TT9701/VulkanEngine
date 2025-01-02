#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Vulkan/Native/Resource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;
class Device;
class Instance;

class MemoryAllocator {
public:
    MemoryAllocator(PhysicalDevice& physicalDevice, Device& device,
                    Instance& instance);

    ~MemoryAllocator();

    CLASS_MOVABLE_ONLY(MemoryAllocator);

public:
    VmaAllocator GetHandle() const { return mHandle; }

private:
    VmaAllocator CreateAllocator();

private:
    PhysicalDevice& mPhysicalDevice;
    Device& mDevice;
    Instance& mInstance;

    VmaAllocator mHandle;
};

class ExternalMemoryPool {
public:
    ExternalMemoryPool(MemoryAllocator& allocator);
    ~ExternalMemoryPool();
    CLASS_MOVABLE_ONLY(ExternalMemoryPool);

public:
    VmaPool GetHandle() const { return mPool; }

private:
    VmaPool CreatePool();

private:
    MemoryAllocator& mAllocator;

    vk::ExportMemoryAllocateInfo mExportMemoryAllocateInfo {
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};

    VmaPool mPool;
};

class AllocatedBase {
public:
    AllocatedBase() = default;
    AllocatedBase(MemoryAllocator& allocator,
                  VmaAllocationCreateInfo const& allocCreateInfo = {});

    CLASS_NO_COPY(AllocatedBase);

    AllocatedBase(AllocatedBase&& other) noexcept;

    const uint8_t* GetData() const;

    VkDeviceMemory GetMemory() const;

    void Flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

    bool Mapped() const;

    uint8_t* Map();

    void Unmap();

    size_t Update(const uint8_t* data, size_t size, size_t offset = 0);

    size_t Update(void const* data, size_t size, size_t offset = 0);

    template <typename T>
    size_t Update(std::vector<T> const& data, size_t offset = 0) {
        return Update(data.data(), data.size() * sizeof(T), offset);
    }

    template <typename T, size_t N>
    size_t Update(std::array<T, N> const& data, size_t offset = 0) {
        return Update(data.data(), data.size() * sizeof(T), offset);
    }
    template <class T>
    size_t ConvertAndUpdate(const T& object, size_t offset = 0) {
        return Update(reinterpret_cast<const uint8_t*>(&object), sizeof(T),
                      offset);
    }

protected:
    virtual void PostCreate(VmaAllocationInfo const& allocationInfo);

    [[nodiscard]] VkBuffer CreateBuffer(VkBufferCreateInfo const& createInfo);

    [[nodiscard]] VkImage CreateImage(VkImageCreateInfo const& createInfo);

    void DestroyBuffer(VkBuffer buffer);

    void DestroyImage(VkImage image);

    void Clear();

protected:
    MemoryAllocator* mAllocator;

    VmaAllocationCreateInfo mAllocCreateInfo {};
    VmaAllocation mAllocation {VK_NULL_HANDLE};
    uint8_t* mMappedData {nullptr};
    bool mCoherent {false};
    bool mPersistent {false};
};

template <typename HandleType, typename MemoryType = vk::DeviceMemory,
          typename ParentType = Resource<HandleType>>
class Allocated : public ParentType, public AllocatedBase {
public:
    CLASS_NO_COPY(Allocated);

    Allocated(Allocated&& other) noexcept;
    Allocated& operator=(Allocated&& other) = default;

    using AllocatedBase::Update;
    using ParentType::GetHandle;
    using ParentType::ParentType;

    Allocated(VulkanContext& context, MemoryAllocator& allocator,
              VmaAllocationCreateInfo const& allocCreateInfo,
              HandleType handle);
};

template <typename HandleType, typename MemoryType, typename ParentType>
Allocated<HandleType, MemoryType, ParentType>::Allocated(
    Allocated&& other) noexcept
    : ParentType {static_cast<ParentType&&>(other)},
      AllocatedBase {static_cast<AllocatedBase&&>(other)} {}

template <typename HandleType, typename MemoryType, typename ParentType>
Allocated<HandleType, MemoryType, ParentType>::Allocated(
    VulkanContext& context, MemoryAllocator& allocator,
    VmaAllocationCreateInfo const& allocCreateInfo, HandleType handle)
    : ParentType(context, handle), AllocatedBase(allocator, allocCreateInfo) {}

}  // namespace IntelliDesign_NS::Vulkan::Core