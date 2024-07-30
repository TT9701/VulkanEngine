#pragma once

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

class VulkanInstance {
public:
    VulkanInstance(::std::span<::std::string> requestedInstanceLayers,
                   ::std::span<::std::string> requestedInstanceExtensions);
    ~VulkanInstance();
    MOVABLE_ONLY(VulkanInstance);

    vk::Instance GetHandle() const { return mInstance; }

private:
    vk::Instance CreateInstance();

private:
    ::std::vector<::std::string> mEnabledInstanceLayers;
    ::std::vector<::std::string> mEnabledInstanceExtensions;
    vk::Instance                 mInstance;
};