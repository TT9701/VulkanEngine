#include "VulkanFrameObjects.hpp"

VulkanFrameObjects::VulkanFrameObjects(
    SharedPtr<VulkanDevice> const& device)
    : pDevice(device) {}