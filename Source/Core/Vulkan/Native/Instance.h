#pragma once

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Utilities/VulkanUtilities.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Instance {
public:
    Instance(::std::span<Type_STLString> requestedInstanceLayers,
             ::std::span<Type_STLString> requestedInstanceExtensions);
    ~Instance();
    MOVABLE_ONLY(Instance);

    vk::Instance GetHandle() const { return mInstance; }

private:
    vk::Instance CreateInstance();

private:
    Type_STLVector<Type_STLString> mEnabledInstanceLayers;
    Type_STLVector<Type_STLString> mEnabledInstanceExtensions;
    vk::Instance mInstance;
};

}  // namespace IntelliDesign_NS::Vulkan::Core