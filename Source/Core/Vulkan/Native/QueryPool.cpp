#include "QueryPool.h"

#include "Core/Vulkan/Manager/VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {
QueryPool::QueryPool(VulkanContext& context) : mContext(context) {
    vk::QueryPoolCreateInfo info {};
    info.setQueryType(vk::QueryType::eTimestamp).setQueryCount(MAX_QUERY_COUNT);
    mPool = mContext.GetDevice()->createQueryPool(info);
}

QueryPool::~QueryPool() {
    mContext.GetDevice()->destroy(mPool);
}

void QueryPool::ResetPool(vk::CommandBuffer cmd, uint32_t count) {
    mValidCount = count;
    cmd.resetQueryPool(mPool, 0, count);
    mCurrentQueryIdx = 0;
}

void QueryPool::BeginRange(vk::CommandBuffer cmd, const char* name,
                           vk::PipelineStageFlagBits2 stage) {
    assert(mCurrentQueryIdx < MAX_QUERY_COUNT);
    mRangeMap.emplace(name, ::std::pair {mCurrentQueryIdx, -1});
    cmd.writeTimestamp2(stage, mPool, mCurrentQueryIdx++);
}

void QueryPool::EndRange(vk::CommandBuffer cmd, const char* name,
                         vk::PipelineStageFlagBits2 stage) {
    assert(mCurrentQueryIdx < MAX_QUERY_COUNT);

    mRangeMap.at(name).second = mCurrentQueryIdx;
    cmd.writeTimestamp2(stage, mPool, mCurrentQueryIdx++);
}

void QueryPool::GetResult() {
    uint32_t count = mCurrentQueryIdx;

    VK_CHECK(mContext.GetDevice()->getQueryPoolResults(
        mPool, 0, count, sizeof(uint64_t) * count, mValues.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait));
}

float QueryPool::ElapsedTime(const char* name) {
    if (!mRangeMap.contains(name))
        return 0.0f;

    float frequency = mContext.GetPhysicalDevice().GetProperties().limits.timestampPeriod;

    auto query = mRangeMap.at(name);

    return (mValues[query.second] - mValues[query.first]) * frequency
         / 1000000.0f;
}

}  // namespace IntelliDesign_NS::Vulkan::Core