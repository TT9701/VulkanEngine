#pragma once

#include "DGCSeqLayout.h"
#include "Pipeline.h"

namespace IntelliDesign_NS::Vulkan::Core {

class DGCSeqRenderLayout {
public:
    template <class TDGCSeqTemplate>
    void CreateLayout(VulkanContext& context, PipelineLayout* pipelineLayout,
                      bool unorderedSequence, bool explicitPreprocess);

    PipelineLayout* GetPipelineLayout() const;

    vk::IndirectCommandsLayoutEXT GetHandle() const;

private:
    PipelineLayout* mPipelineLayout {nullptr};

    Type_UniquePtr<DGCSeqLayout> mLayout;
};

template <class TDGCSeqTemplate>
void DGCSeqRenderLayout::CreateLayout(VulkanContext& context,
                                      PipelineLayout* pipelineLayout,
                                      bool unorderedSequence,
                                      bool explicitPreprocess) {
    mPipelineLayout = pipelineLayout;

    mLayout = IntelliDesign_NS::Vulkan::Core::CreateLayout<TDGCSeqTemplate>(
        context, pipelineLayout->GetHandle(), unorderedSequence,
        explicitPreprocess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core