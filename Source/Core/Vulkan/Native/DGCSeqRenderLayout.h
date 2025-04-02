#pragma once

#include "DGCSeqLayout.h"
#include "Pipeline.h"

namespace IntelliDesign_NS::Vulkan::Core {

class DGCSeqRenderLayout {
public:
    template <class TDGCSeqTemplate>
    void CreateLayout(VulkanContext& context, PipelineLayout* pipelineLayout,
                      bool explicitPreprocess);

    PipelineLayout const* GetPipelineLayout() const;

    vk::IndirectCommandsLayoutEXT GetHandle() const;

private:
    PipelineLayout* mPipelineLayout {nullptr};

    Type_UniquePtr<DGCSeqLayout> mLayout;
};

template <class TDGCSeqTemplate>
void DGCSeqRenderLayout::CreateLayout(VulkanContext& context,
                                      PipelineLayout* pipelineLayout,
                                      bool explicitPreprocess) {
    mPipelineLayout = pipelineLayout;

    mLayout = Core::CreateLayout<TDGCSeqTemplate>(
        context, pipelineLayout->GetHandle(), explicitPreprocess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core