#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

class VulkanDevice;
class VulkanFence;
class VulkanSemaphore;

class VulkanFrameObjects {
    USING_TEMPLATE_SHARED_PTR_TYPE(Type_SPInstance);

public:
    VulkanFrameObjects(Type_SPInstance<VulkanDevice> const& device);
    ~VulkanFrameObjects() = default;
    MOVABLE_ONLY(VulkanFrameObjects);

private:
    Type_SPInstance<VulkanDevice> pDevice;

    Type_SPInstance<VulkanFence> mRenderFence;
    Type_SPInstance<VulkanSemaphore> mReady4RenderSemaphore;
    Type_SPInstance<VulkanSemaphore> mReady4PresentSemaphore;
};