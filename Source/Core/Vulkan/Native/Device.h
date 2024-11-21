#pragma once

#include "CommandBuffer.h"
#include "CommandPool.h"
#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "DebugUtils.h"
#include "PhysicalDevice.h"
#include "Queue.h"
#include "FencePool.h"
#include "VulkanResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Device : public VulkanResource<vk::Device> {
public:
    Device(PhysicalDevice& physicalDevice, vk::SurfaceKHR surface,
           UniquePtr<DebugUtils>&& debug_utils,
           std::unordered_map<const char*, bool> requested_extensions = {});

    ~Device() override;
    CLASS_NO_COPY_MOVE(Device);

    PhysicalDevice const& get_gpu() const;

    DebugUtils const& get_debug_utils() const;

    Queue const& get_queue(uint32_t queue_family_index,
                                         uint32_t queue_index) const;

    Queue const& get_queue_by_flags(vk::QueueFlags queue_flags,
                                                  uint32_t queue_index) const;

    Queue const& get_queue_by_present(uint32_t queue_index) const;

    Queue const& get_suitable_graphics_queue() const;

    bool is_extension_supported(std::string const& extension) const;

    bool is_enabled(std::string const& extension) const;

    uint32_t get_queue_family_index(vk::QueueFlagBits queue_flag) const;

    HPPCommandPool& get_command_pool();

    // std::pair<vk::Image, vk::DeviceMemory> create_image(
    //     vk::Format format, vk::Extent2D const& extent, uint32_t mip_levels,
    //     vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties) const;
    //
    // void copy_buffer(vkb::core::BufferCpp& src, vkb::core::BufferCpp& dst,
    //                  vk::Queue queue,
    //                  vk::BufferCopy* copy_region = nullptr) const;

    vk::CommandBuffer create_command_buffer(vk::CommandBufferLevel level,
                                            bool begin = false) const;

    void flush_command_buffer(
        vk::CommandBuffer command_buffer, vk::Queue queue, bool free = true,
        vk::Semaphore signalSemaphore = VK_NULL_HANDLE) const;

    FencePool& get_fence_pool();

private:
    PhysicalDevice const& gpu;

    vk::SurfaceKHR surface {nullptr};

    UniquePtr<DebugUtils> debug_utils;

    std::vector<const char*> enabled_extensions {};

    std::vector<std::vector<Queue>> queues;

    UniquePtr<HPPCommandPool> command_pool;

    UniquePtr<FencePool> fence_pool;
};

}  // namespace IntelliDesign_NS::Vulkan::Core