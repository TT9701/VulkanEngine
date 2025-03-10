#include "DGCSeqLayout.h"

namespace IntelliDesign_NS::Vulkan::Core {

DGCSeqLayout::DGCSeqLayout(VulkanContext& context) : mContext(context) {}

DGCSeqLayout::~DGCSeqLayout() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::IndirectCommandsLayoutEXT DGCSeqLayout::GetHandle() const {
    return mHandle;
}

}  // namespace IntelliDesign_NS::Vulkan::Core