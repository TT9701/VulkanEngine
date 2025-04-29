#include "Profiler.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Core {
TracyProfiler::TracyProfiler(Vulkan::Core::VulkanContext& context) {
    mTracyCtx = TracyVkContextHostCalibrated(
        context.GetInstance().GetHandle(),
        context.GetPhysicalDevice().GetHandle(),
        context.GetDevice().GetHandle(), vkGetInstanceProcAddr,
        vkGetDeviceProcAddr);
}

TracyProfiler::~TracyProfiler() {
    TracyVkDestroy(mTracyCtx);
}

}  // namespace IntelliDesign_NS::Core