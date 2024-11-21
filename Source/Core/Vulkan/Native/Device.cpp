#include "Device.h"

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/VulkanUtilities.h"
#include "PhysicalDevice.h"

namespace IntelliDesign_NS::Vulkan::Core {

Device::Device(PhysicalDevice& physicalDevice, vk::SurfaceKHR surface,
               UniquePtr<DebugUtils>&& debug_utils,
               std::unordered_map<const char*, bool> requested_extensions)
    : VulkanResource(VK_NULL_HANDLE, this),
      debug_utils {std::move(debug_utils)},
      gpu {physicalDevice} {
    DBG_LOG_INFO(::std::string {"Selected GPU: "}
                 + gpu.GetProperties().deviceName.data());

    // Prepare the device queues
    auto queue_family_properties = gpu.GetQueueFamilyProperties();
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos(
        queue_family_properties.size());
    std::vector<std::vector<float>> queue_priorities(
        queue_family_properties.size());

    for (uint32_t queue_family_index = 0U;
         queue_family_index < queue_family_properties.size();
         ++queue_family_index) {
        vk::QueueFamilyProperties const& queue_family_property =
            queue_family_properties[queue_family_index];

        if (gpu.HasHighPriorityGraphicsQueue()) {
            uint32_t graphics_queue_family =
                get_queue_family_index(vk::QueueFlagBits::eGraphics);
            if (graphics_queue_family == queue_family_index) {
                queue_priorities[queue_family_index].reserve(
                    queue_family_property.queueCount);
                queue_priorities[queue_family_index].push_back(1.0f);
                for (uint32_t i = 1; i < queue_family_property.queueCount;
                     i++) {
                    queue_priorities[queue_family_index].push_back(0.5f);
                }
            } else {
                queue_priorities[queue_family_index].resize(
                    queue_family_property.queueCount, 0.5f);
            }
        } else {
            queue_priorities[queue_family_index].resize(
                queue_family_property.queueCount, 0.5f);
        }

        vk::DeviceQueueCreateInfo& queue_create_info =
            queue_create_infos[queue_family_index];

        queue_create_info.queueFamilyIndex = queue_family_index;
        queue_create_info.queueCount = queue_family_property.queueCount;
        queue_create_info.pQueuePriorities =
            queue_priorities[queue_family_index].data();
    }

    // Check extensions to enable Vma Dedicated Allocation
    bool can_get_memory_requirements =
        is_extension_supported(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    bool has_dedicated_allocation =
        is_extension_supported(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

    if (can_get_memory_requirements && has_dedicated_allocation) {
        enabled_extensions.push_back(
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        enabled_extensions.push_back(
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

        DBG_LOG_INFO("Dedicated Allocation enabled");
    }

    // For performance queries, we also use host query reset since queryPool resets cannot
    // live in the same command buffer as beginQuery
    if (is_extension_supported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)
        && is_extension_supported(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME)) {
        auto perf_counter_features = physicalDevice.GetExtensionFeatures<
            vk::PhysicalDevicePerformanceQueryFeaturesKHR>();
        auto host_query_reset_features = physicalDevice.GetExtensionFeatures<
            vk::PhysicalDeviceHostQueryResetFeatures>();

        if (perf_counter_features.performanceCounterQueryPools
            && host_query_reset_features.hostQueryReset) {
            physicalDevice
                .AddExtensionFeatures<
                    vk::PhysicalDevicePerformanceQueryFeaturesKHR>()
                .performanceCounterQueryPools = true;
            physicalDevice
                .AddExtensionFeatures<
                    vk::PhysicalDeviceHostQueryResetFeatures>()
                .hostQueryReset = true;
            enabled_extensions.push_back(
                VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);
            enabled_extensions.push_back(
                VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
            DBG_LOG_INFO("Performance query enabled");
        }
    }

    // Check that extensions are supported before trying to create the device
    std::vector<const char*> unsupported_extensions {};
    for (auto& extension : requested_extensions) {
        if (is_extension_supported(extension.first)) {
            enabled_extensions.emplace_back(extension.first);
        } else {
            unsupported_extensions.emplace_back(extension.first);
        }
    }

    if (!enabled_extensions.empty()) {
        DBG_LOG_INFO("HPPDevice supports the following requested extensions:");
        for (auto& extension : enabled_extensions) {
            DBG_LOG_INFO(::std::string {"  \t"} + extension);
        }
    }

    if (!unsupported_extensions.empty()) {
        auto error = false;
        for (auto& extension : unsupported_extensions) {
            auto extension_is_optional = requested_extensions[extension];
            if (extension_is_optional) {
                DBG_LOG_INFO(::std::string {"Optional device extension {} not "
                                            "available, some features "
                                            "may be disabled"}
                             + extension);
            } else {
                DBG_LOG_INFO(::std::string {"Required device extension {} not "
                                            "available, cannot run"}
                             + extension);
                error = true;
            }
        }

        if (error) {
            throw ::std::runtime_error("Extensions not present");
        }
    }

    vk::DeviceCreateInfo create_info(
        {}, queue_create_infos, {}, enabled_extensions,
        &physicalDevice.GetMutableRequestedFeatures());

    // Latest requested feature will have the pNext's all set up for device creation.
    create_info.pNext = physicalDevice.GetExtensionFeatureChain();

    SetHandle(gpu.GetHandle().createDevice(create_info));

    queues.resize(queue_family_properties.size());

    for (uint32_t queue_family_index = 0U;
         queue_family_index < queue_family_properties.size();
         ++queue_family_index) {
        vk::QueueFamilyProperties const& queue_family_property =
            queue_family_properties[queue_family_index];

        vk::Bool32 present_supported =
            gpu.GetHandle().getSurfaceSupportKHR(queue_family_index, surface);

        for (uint32_t queue_index = 0U;
             queue_index < queue_family_property.queueCount; ++queue_index) {
            queues[queue_family_index].emplace_back(
                *this, queue_family_index, queue_family_property,
                present_supported, queue_index);
        }
    }

    // vkb::allocated::init(*this);

    command_pool = MakeUnique<HPPCommandPool>(
        *this,
        get_queue_by_flags(
            vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute, 0)
            .GetFamilyIndex());
    fence_pool = MakeUnique<FencePool>(*this);
}

Device::~Device() {
    command_pool.reset();
    fence_pool.reset();

    // vkb::allocated::shutdown();

    if (GetHandle()) {
        GetHandle().destroy();
    }
}

PhysicalDevice const& Device::get_gpu() const {
    return gpu;
}

DebugUtils const& Device::get_debug_utils() const {
    return *debug_utils;
}

Queue const& Device::get_queue(uint32_t queue_family_index,
                               uint32_t queue_index) const {
    return queues[queue_family_index][queue_index];
}

Queue const& Device::get_queue_by_flags(vk::QueueFlags queue_flags,
                                        uint32_t queue_index) const {
    for (size_t queue_family_index = 0U; queue_family_index < queues.size();
         ++queue_family_index) {
        Queue const& first_queue = queues[queue_family_index][0];

        vk::QueueFlags queue_flags = first_queue.GetProperties().queueFlags;
        uint32_t queue_count = first_queue.GetProperties().queueCount;

        if (((queue_flags & queue_flags) == queue_flags)
            && queue_index < queue_count) {
            return queues[queue_family_index][queue_index];
        }
    }

    throw std::runtime_error("Queue not found");
}

Queue const& Device::get_queue_by_present(uint32_t queue_index) const {
    for (uint32_t queue_family_index = 0U; queue_family_index < queues.size();
         ++queue_family_index) {
        Queue const& first_queue = queues[queue_family_index][0];

        uint32_t queue_count = first_queue.GetProperties().queueCount;

        if (first_queue.SupportPresent() && queue_index < queue_count) {
            return queues[queue_family_index][queue_index];
        }
    }

    throw std::runtime_error("Queue not found");
}

Queue const& Device::get_suitable_graphics_queue() const {
    for (size_t queue_family_index = 0U; queue_family_index < queues.size();
         ++queue_family_index) {
        Queue const& first_queue = queues[queue_family_index][0];

        uint32_t queue_count = first_queue.GetProperties().queueCount;

        if (first_queue.SupportPresent() && 0 < queue_count) {
            return queues[queue_family_index][0];
        }
    }

    return get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
}

bool Device::is_extension_supported(std::string const& extension) const {
    return gpu.IsExtensionSupported(extension);
}

bool Device::is_enabled(std::string const& extension) const {
    return std::ranges::find_if(enabled_extensions,
                                [extension](const char* enabled_extension) {
                                    return extension == enabled_extension;
                                })
        != enabled_extensions.end();
}

uint32_t Device::get_queue_family_index(vk::QueueFlagBits queue_flag) const {
    const auto& queue_family_properties = gpu.GetQueueFamilyProperties();

    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if (queue_flag & vk::QueueFlagBits::eCompute) {
        for (uint32_t i = 0;
             i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
            if ((queue_family_properties[i].queueFlags & queue_flag)
                && !(queue_family_properties[i].queueFlags
                     & vk::QueueFlagBits::eGraphics)) {
                return i;
                break;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if (queue_flag & vk::QueueFlagBits::eTransfer) {
        for (uint32_t i = 0;
             i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
            if ((queue_family_properties[i].queueFlags & queue_flag)
                && !(queue_family_properties[i].queueFlags
                     & vk::QueueFlagBits::eGraphics)
                && !(queue_family_properties[i].queueFlags
                     & vk::QueueFlagBits::eCompute)) {
                return i;
                break;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested
    // flags
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
        if (queue_family_properties[i].queueFlags & queue_flag) {
            return i;
            break;
        }
    }

    throw std::runtime_error("Could not find a matching queue family index");
}

HPPCommandPool& Device::get_command_pool() {
    return *command_pool;
}

vk::CommandBuffer Device::create_command_buffer(vk::CommandBufferLevel level,
                                                bool begin) const {
    assert(command_pool && "No command pool exists in the device");

    vk::CommandBuffer command_buffer =
        GetHandle()
            .allocateCommandBuffers({command_pool->GetHandle(), level, 1})
            .front();

    // If requested, also start recording for the new command buffer
    if (begin) {
        command_buffer.begin(vk::CommandBufferBeginInfo());
    }

    return command_buffer;
}

void Device::flush_command_buffer(vk::CommandBuffer command_buffer,
                                  vk::Queue queue, bool free,
                                  vk::Semaphore signalSemaphore) const {
    if (!command_buffer) {
        return;
    }

    command_buffer.end();

    vk::SubmitInfo submit_info({}, {}, command_buffer);
    if (signalSemaphore) {
        submit_info.setSignalSemaphores(signalSemaphore);
    }

    // Create fence to ensure that the command buffer has finished executing
    vk::Fence fence = GetHandle().createFence({});

    // Submit to the queue
    queue.submit(submit_info, fence);

    // Wait for the fence to signal that command buffer has finished executing
    vk::Result result =
        GetHandle().waitForFences(fence, true, DEFAULT_FENCE_TIME_OUT);
    if (result != vk::Result::eSuccess) {
        DBG_LOG_INFO(::std::string {"Detected Vulkan error: "}
                     + vk::to_string(result));
        abort();
    }

    GetHandle().destroyFence(fence);

    if (command_pool && free) {
        GetHandle().freeCommandBuffers(command_pool->GetHandle(),
                                        command_buffer);
    }
}

FencePool& Device::get_fence_pool() {
    return *fence_pool;
}

}  // namespace IntelliDesign_NS::Vulkan::Core