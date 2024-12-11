#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

class Surface {
public:
    Surface(Instance& instance, const SDLWindow& window);
    ~Surface();
    CLASS_MOVABLE_ONLY(Surface);

public:
    vk::SurfaceKHR const& GetHandle() const { return mHandle; }

private:
    Instance& mInstance;
    vk::SurfaceKHR mHandle;
};

}  // namespace IntelliDesign_NS::Vulkan::Core