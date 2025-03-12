#include "DGCSeqRenderLayout.h"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineLayout* DGCSeqRenderLayout::GetPipelineLayout() const {
    VE_ASSERT(mPipelineLayout != nullptr, "Pipeline layout is null.");
    return mPipelineLayout;
}

vk::IndirectCommandsLayoutEXT DGCSeqRenderLayout::GetHandle() const {
    VE_ASSERT(mLayout != nullptr, "Layout is null.");
    return mLayout->GetHandle();
}
}  // namespace IntelliDesign_NS::Vulkan::Core