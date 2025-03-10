#include "DGCSeqRenderLayout.h"

namespace IntelliDesign_NS::Vulkan::Core {

ShaderProgram* DGCSeqRenderLayout::GetShaderProgram() const {
    VE_ASSERT(mShaderProgram != nullptr, "Shader program is null.");
    return mShaderProgram;
}

vk::IndirectCommandsLayoutEXT DGCSeqRenderLayout::GetHandle() const {
    VE_ASSERT(mLayout != nullptr, "Layout is null.");
    return mLayout->GetHandle();
}
}  // namespace IntelliDesign_NS::Vulkan::Core