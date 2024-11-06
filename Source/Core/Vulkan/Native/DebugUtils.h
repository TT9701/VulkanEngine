#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

class DebugUtils {
public:
    DebugUtils(Instance* instance);
    ~DebugUtils();
    MOVABLE_ONLY(DebugUtils);

public:
    vk::DebugUtilsMessengerEXT GetHandle() const { return mDebugMessenger; }

private:
    vk::DebugUtilsMessengerEXT CreateDebugMessenger();

private:
    Instance* pInstance;
    vk::DebugUtilsMessengerEXT mDebugMessenger;
};

}  // namespace IntelliDesign_NS::Vulkan::Core