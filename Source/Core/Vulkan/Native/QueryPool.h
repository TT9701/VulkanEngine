#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/MemoryPool.h"

constexpr uint32_t MAX_QUERY_COUNT = 128;

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

class QueryPool {
public:
    QueryPool(VulkanContext& context);

    ~QueryPool();

    void ResetPool(vk::CommandBuffer cmd, uint32_t count);

    void BeginRange(vk::CommandBuffer cmd, const char* name,
                    vk::PipelineStageFlagBits2 stage =
                        vk::PipelineStageFlagBits2::eTopOfPipe);

    void EndRange(vk::CommandBuffer cmd, const char* name,
                  vk::PipelineStageFlagBits2 stage =
                      vk::PipelineStageFlagBits2::eBottomOfPipe);

    void GetResult();

    float ElapsedTime(const char* name);

private:
    VulkanContext& mContext;

    vk::QueryPool mPool {};
    uint32_t mValidCount {};
    ::std::array<uint64_t, MAX_QUERY_COUNT> mValues {};
    uint32_t mCurrentQueryIdx {0};
    Type_STLUnorderedMap_String<::std::pair<uint32_t, uint32_t>> mRangeMap {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core