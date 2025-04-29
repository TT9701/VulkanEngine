#pragma once

#include <vulkan/vulkan.h>
#include "tracy/TracyVulkan.hpp"

namespace tracy {
class VkCtx;
}

namespace IntelliDesign_NS::Vulkan::Core {
class VulkanContext;
}

namespace IntelliDesign_NS::Core {

class TracyProfiler {
public:
    TracyProfiler(Vulkan::Core::VulkanContext& context);

    ~TracyProfiler();

    tracy::VkCtx* GetTracyCtx() const { return mTracyCtx; }

private:
    tracy::VkCtx* mTracyCtx {nullptr};
};

}  // namespace IntelliDesign_NS::Core