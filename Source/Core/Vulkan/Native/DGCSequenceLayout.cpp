#include "DGCSequenceLayout.h"

namespace IntelliDesign_NS::Vulkan::Core {

SequenceLayout::SequenceLayout(VulkanContext& context) : mContext(context) {}

SequenceLayout::~SequenceLayout() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::IndirectCommandsLayoutEXT SequenceLayout::GetHandle() const {
    return mHandle;
}

}  // namespace IntelliDesign_NS::Vulkan::Core