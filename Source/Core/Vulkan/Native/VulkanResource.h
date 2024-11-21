#pragma once

#include <vulkan/vulkan.hpp>

#include "Device.h"

namespace IntelliDesign_NS::Vulkan::Core {
class Device;

template <class Handle>
class VulkanResource {
public:
    using ObjectType = vk::ObjectType;

    VulkanResource(Handle handle = VK_NULL_HANDLE, Device* device = nullptr);

    VulkanResource(const VulkanResource&) = delete;
    VulkanResource& operator=(const VulkanResource&) = delete;

    VulkanResource(VulkanResource&& other) noexcept;
    VulkanResource& operator=(VulkanResource&& other) noexcept;

    virtual ~VulkanResource() = default;

    const std::string& GetDebugName() const;
    Device& GetDevice();
    Device const& GetDevice() const;
    Handle& GetHandle();
    const Handle& GetHandle() const;
    uint64_t GetHandle_u64() const;
    ObjectType GetObjectType() const;
    bool HasDevice() const;
    bool HasHandle() const;
    void SetDebugName(const std::string& name);
    void SetHandle(Handle hdl);

private:
    std::string mDebugName;
    Device* pDevice;
    Handle mHandle;
};

template <class Handle>
VulkanResource<Handle>::VulkanResource(Handle handle, Device* device)
    : pDevice(device), mHandle(handle) {}

template <class Handle>
VulkanResource<Handle>::VulkanResource(VulkanResource&& other) noexcept
    : mDebugName(std::exchange(other.mDebugName, {})),
      pDevice(std::exchange(other.pDevice, {})),
      mHandle(std::exchange(other.mHandle, {})) {}

template <class Handle>
VulkanResource<Handle>& VulkanResource<Handle>::operator=(
    VulkanResource&& other) noexcept {
    mDebugName = std::exchange(other.mDebugName, {});
    pDevice = std::exchange(other.pDevice, {});
    mHandle = std::exchange(other.mHandle, {});
    return *this;
}

template <class Handle>
const std::string& VulkanResource<Handle>::GetDebugName() const {
    return mDebugName;
}

template <class Handle>
Device& VulkanResource<Handle>::GetDevice() {
    assert(pDevice && "Device handle not set");
    return *pDevice;
}

template <class Handle>
Device const& VulkanResource<Handle>::GetDevice() const {
    assert(pDevice && "Device handle not set");
    return *pDevice;
}

template <class Handle>
Handle& VulkanResource<Handle>::GetHandle() {
    return mHandle;
}

template <class Handle>
const Handle& VulkanResource<Handle>::GetHandle() const {
    return mHandle;
}

template <class Handle>
uint64_t VulkanResource<Handle>::GetHandle_u64() const {
    using UintHandle = std::conditional_t<sizeof(Handle) == sizeof(uint32_t),
                                          uint32_t, uint64_t>;

    return static_cast<uint64_t>(
        *reinterpret_cast<UintHandle const*>(&mHandle));
}

template <class Handle>
typename VulkanResource<Handle>::ObjectType
VulkanResource<Handle>::GetObjectType() const {
    return Handle::objectType;
}

template <class Handle>
bool VulkanResource<Handle>::HasDevice() const {
    return pDevice != nullptr;
}

template <class Handle>
bool VulkanResource<Handle>::HasHandle() const {
    return mHandle != VK_NULL_HANDLE;
}

template <class Handle>
void VulkanResource<Handle>::SetDebugName(const std::string& name) {
    mDebugName = name;
    if (pDevice && !mDebugName.empty()) {
        GetDevice().SetObjectName(GetHandle(), mDebugName.c_str());
    }
}

template <class Handle>
void VulkanResource<Handle>::SetHandle(Handle hdl) {
    mHandle = hdl;
}

}  // namespace IntelliDesign_NS::Vulkan::Core
