#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"

class SDLWindow;

namespace IntelliDesign_NS::Vulkan::Core {

class Instance;

class Surface {
public:
    Surface(Instance* instance, const SDLWindow* window);
    ~Surface();
    CLASS_MOVABLE_ONLY(Surface);

public:
    vk::SurfaceKHR const& GetHandle() const { return mSurface; }

private:
    vk::SurfaceKHR CreateSurface(const SDLWindow* window) const;

private:
    Instance* pInstance;
    vk::SurfaceKHR mSurface;
};

}  // namespace IntelliDesign_NS::Vulkan::Core