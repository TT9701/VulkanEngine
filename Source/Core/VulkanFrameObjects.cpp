#include "VulkanFrameObjects.hpp"

VulkanFrameObjects::VulkanFrameObjects(
    Type_SPInstance<VulkanDevice> const& device)
    : pDevice(device) {}