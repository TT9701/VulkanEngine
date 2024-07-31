#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanDevice;
class VulkanFence;
class VulkanSemaphore;

class VulkanFrameObjects {
public:
    VulkanFrameObjects(VulkanDevice* device);
    ~VulkanFrameObjects() = default;
    MOVABLE_ONLY(VulkanFrameObjects);

private:
    VulkanDevice* pDevice;

    SharedPtr<VulkanFence>     mRenderFence;
    SharedPtr<VulkanSemaphore> mReady4RenderSemaphore;
    SharedPtr<VulkanSemaphore> mReady4PresentSemaphore;
};