#pragma once

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

class VulkanInstance {
public:
    VulkanInstance(
        ::std::vector<::std::string> const& requestedInstanceLayers,
        ::std::vector<::std::string> const& requestedInstanceExtensions);
    ~VulkanInstance();
    MOVABLE_ONLY(VulkanInstance);

    vk::Instance const& GetHandle() const { return mInstance; }

private:
    vk::Instance CreateInstance();

private:
    ::std::vector<::std::string> mEnabledInstanceLayers;
    ::std::vector<::std::string> mEnabledInstanceExtensions;
    vk::Instance mInstance;
};