#pragma once

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/VulkanUtilities.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance {
public:
    Instance(::std::span<::std::string> requestedInstanceLayers,
                   ::std::span<::std::string> requestedInstanceExtensions);
    ~Instance();
    MOVABLE_ONLY(Instance);

    vk::Instance GetHandle() const { return mInstance; }

private:
    vk::Instance CreateInstance();

private:
    ::std::vector<::std::string> mEnabledInstanceLayers;
    ::std::vector<::std::string> mEnabledInstanceExtensions;
    vk::Instance mInstance;
};

}  // namespace IntelliDesign_NS::Vulkan::Core