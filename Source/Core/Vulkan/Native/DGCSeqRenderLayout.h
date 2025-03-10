#pragma once

#include "DGCSeqLayout.h"
#include "Pipeline.h"

namespace IntelliDesign_NS::Vulkan::Core {

class DGCSeqRenderLayout {
public:
    template <class TDGCSeqTemplate>
    void CreateLayout(VulkanContext& context, PipelineLayout* pipelineLayout,
                      bool unorderedSequence, bool explicitPreprocess);

    template <class TDGCSeqTemplate>
    void CreateLayout(VulkanContext& context, ShaderProgram* shaderProgram,
                      bool unorderedSequence, bool explicitPreprocess);

    ShaderProgram* GetShaderProgram() const;

    vk::IndirectCommandsLayoutEXT GetHandle() const;

private:
    ShaderProgram* mShaderProgram {nullptr};

    Type_UniquePtr<DGCSeqLayout> mLayout;
};

template <class TDGCSeqTemplate>
void DGCSeqRenderLayout::CreateLayout(VulkanContext& context,
                                      PipelineLayout* pipelineLayout,
                                      bool unorderedSequence,
                                      bool explicitPreprocess) {
    mShaderProgram = &pipelineLayout->GetShaderProgram();

    mLayout = IntelliDesign_NS::Vulkan::Core::CreateLayout<TDGCSeqTemplate>(
        context, pipelineLayout->GetHandle(), unorderedSequence,
        explicitPreprocess);
}

template <class TDGCSeqTemplate>
void DGCSeqRenderLayout::CreateLayout(VulkanContext& context,
                                      ShaderProgram* shaderProgram,
                                      bool unorderedSequence,
                                      bool explicitPreprocess) {
    mShaderProgram = shaderProgram;

    mLayout = IntelliDesign_NS::Vulkan::Core::CreateLayout<TDGCSeqTemplate>(
        context, shaderProgram->GetCombinedDescLayoutHandles(),
        shaderProgram->GetPCRanges()[0], unorderedSequence, explicitPreprocess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core