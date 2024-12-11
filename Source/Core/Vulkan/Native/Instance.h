#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Utilities/VulkanUtilities.h"

// TODO: extension 可用性判断

namespace IntelliDesign_NS::Vulkan::Core {

class PhysicalDevice;

class Instance {
public:
    Instance(::std::span<Type_STLString> requestedInstanceLayers,
             ::std::span<Type_STLString> requestedInstanceExtensions);
    ~Instance();
    CLASS_NO_COPY_MOVE(Instance);

public:
    vk::Instance GetHandle() const { return mHandle; }

    ::std::span<Type_STLString> GetExtensions();

    PhysicalDevice& GetSuitableGPU(vk::SurfaceKHR surface) const;

private:
    void QueyGPUs();

private:
    Type_STLVector<Type_STLString> mEnabledLayers;
    Type_STLVector<Type_STLString> mEnabledExtensions;
    vk::Instance mHandle;
    Type_STLVector<UniquePtr<PhysicalDevice>> mGPUs;
};

}  // namespace IntelliDesign_NS::Vulkan::Core