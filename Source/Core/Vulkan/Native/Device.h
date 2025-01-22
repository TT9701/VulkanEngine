#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Queue.h"

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;
class Surface;

/**
 * @brief 
 */
class Device {
public:
    /**
     * @brief 
     * @param physicalDevice 
     * @param surface 
     * @param requestedExtensions 
     */
    Device(PhysicalDevice& physicalDevice, Surface& surface,
           ::std::span<Type_STLString> requestedExtensions = {});

    /**
     * @brief 
     */
    ~Device();

    /**
     *
     */
    CLASS_MOVABLE_ONLY(Device);

public:
    /**
     * @brief 
     * @return 
     */
    vk::Device GetHandle() const;

    /**
     * @brief 
     * @return 
     */
    vk::Device const* operator->() const;

    /**
     * @brief 
     * @param queueFlag 
     * @return 
     */
    uint32_t GetQueueFamilyIndex(vk::QueueFlagBits queueFlag) const;

    /**
     * @brief 
     * @return 
     */
    Type_STLUnorderedMap<vk::QueueFlagBits, uint32_t> const&
    GetQueueFamilyIndices() const;

    /**
     * @brief 
     * @param extension 
     * @return 
     */
    bool IsExtensionSupported(const char* extension) const;

    /**
     * @brief 
     * @param extension 
     * @return 
     */
    bool IsExtensionEnabled(const char* extension) const;

    /**
     * @brief 
     * @param familyIndex 
     * @param index 
     * @return 
     */
    Queue const& GetQueue(uint32_t familyIndex, uint32_t index) const;

    /**
     * @brief 
     * @tparam VkCppHandle 
     * @param handle 
     * @param name 
     */
    template <class VkCppHandle>
    void SetObjectName(VkCppHandle handle, const char* name);

private:
    /**
     * @brief 
     * @param queueFlag 
     * @return 
     */
    uint32_t GetFamilyIndex(vk::QueueFlagBits queueFlag) const;

    /**
     * @brief 
     * @param requestedExtensions 
     * @return 
     */
    vk::Device CreateDevice(std::span<Type_STLString> requestedExtensions);

    /**
     * @brief 
     * @param surface 
     */
    void CreateQueues(Surface& surface);

private:
    PhysicalDevice& mPhysicalDevice;

    Type_STLVector<Type_STLString> enabledExtensions {};

    vk::Device mHandle;

    Type_STLVector<Type_STLVector<Queue>> mQueues {};
    Type_STLUnorderedMap<vk::QueueFlagBits, uint32_t> mQueueFamilyIndices;
};

}  // namespace IntelliDesign_NS::Vulkan::Core

namespace IntelliDesign_NS::Vulkan::Core {

// TODO: template requirements
template <class VkCppHandle>
void Device::SetObjectName(VkCppHandle handle, const char* name) {
#ifndef NDEBUG
    vk::DebugUtilsObjectNameInfoEXT info {};
    using CType = typename VkCppHandle::CType;
    info.setObjectHandle((uint64_t)(CType)handle)
        .setObjectType(VkCppHandle::objectType)
        .setPObjectName(name);
    mHandle.setDebugUtilsObjectNameEXT(info);
#endif
}
}  // namespace IntelliDesign_NS::Vulkan::Core