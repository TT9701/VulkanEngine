#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class CommandBuffer;
class Device;

class Queue {
public:
    /**
     * @brief 
     * @param device 
     * @param family_index 
     * @param properties 
     * @param canPresent 
     * @param index 
     */
    Queue(Device& device, uint32_t family_index,
          vk::QueueFamilyProperties const& properties, vk::Bool32 canPresent,
          uint32_t index);

    ~Queue() = default;

public:
    /**
     * @brief 
     * @return 
     */
    vk::Queue GetHandle() const;

    /**
     * @brief 
     * @return 
     */
    uint32_t GetFamilyIndex() const;

    /**
     * @brief 
     * @return 
     */
    uint32_t GetIndex() const;

    /**
     * @brief 
     * @return 
     */
    vk::QueueFamilyProperties const& GetFamilyProperties() const;

    /**
     * @brief 
     * @return 
     */
    vk::Bool32 SupportPresent() const;

    /**
     * @brief 
     * @param cmd 
     */
    void Submit(CommandBuffer const& cmd, vk::Fence) const;

    /**
     * @brief 
     * @param info 
     * @return 
     */
    vk::Result Present(vk::PresentInfoKHR const& info) const;

private:
    Device& mDevice;
    vk::Queue mHandle;
    uint32_t mFamilyIndex {0};
    uint32_t mIndex {0};
    vk::Bool32 mCanPresent {false};
    vk::QueueFamilyProperties2 mProperties;
};

}  // namespace IntelliDesign_NS::Vulkan::Core