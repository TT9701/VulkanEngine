#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

class DebugUtils {
public:
    DebugUtils(Instance& instance);
    ~DebugUtils();
    CLASS_MOVABLE_ONLY(DebugUtils);

public:
    vk::DebugUtilsMessengerEXT GetHandle() const { return mHandle; }

private:
    Instance& mInstance;
    vk::DebugUtilsMessengerEXT mHandle;
};

}  // namespace IntelliDesign_NS::Vulkan::Core