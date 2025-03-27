#include "DrawCallMetaData.h"

#include "Core/Vulkan/Manager/RenderResourceManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

void DrawCallMetaData<DrawCallMetaDataType::ClearColorImage>::
    UpdateRenderResource(RenderResourceManager& manager, Type_STLString name) {
    auto resource = manager[name.c_str()].GetTexHandle();
    image = resource;
}

void DrawCallMetaData<DrawCallMetaDataType::ClearColorImage>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.clearColorImage(image, layout, clearValue, ranges);
}

void DrawCallMetaData<DrawCallMetaDataType::ClearDepthStencilImage>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.clearDepthStencilImage(image, layout, clearValue, ranges);
}

void DrawCallMetaData<DrawCallMetaDataType::ResetBuffer>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.fillBuffer(buffer, offset, size, 0);
}

void DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>::RecordCmds(
    vk::CommandBuffer cmd) const {
    vk::DependencyInfo depInfo {};
    // TODO: vk::DependencyFlags
    if (memBarriers.has_value())
        depInfo.setMemoryBarriers(memBarriers.value());
    if (bufBarriers.has_value())
        depInfo.setBufferMemoryBarriers(bufBarriers.value());
    if (imgBarriers.has_value())
        depInfo.setImageMemoryBarriers(imgBarriers.value());

    cmd.pipelineBarrier2(depInfo);
}

void DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.beginRendering(info);
}

void DrawCallMetaData<DrawCallMetaDataType::Viewport>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.setViewport(firstViewport, viewports);
}

void DrawCallMetaData<DrawCallMetaDataType::Scissor>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.setScissor(firstScissor, scissors);
}

void DrawCallMetaData<DrawCallMetaDataType::Pipeline>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.bindPipeline(bindPoint, pipeline);
}

void DrawCallMetaData<DrawCallMetaDataType::DescriptorBuffer>::RecordCmds(
    vk::CommandBuffer cmd) const {
    auto bufferCount = addresses.size();
    Type_STLVector<vk::DescriptorBufferBindingInfoEXT> infos(bufferCount);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        infos[i]
            .setUsage(vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT
                      | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT)
            .setAddress(addresses[i]);
    }

    cmd.bindDescriptorBuffersEXT(infos);
}

void DrawCallMetaData<DrawCallMetaDataType::DescriptorSet>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.setDescriptorBufferOffsetsEXT(bindPoint, layout, firstSet,
                                      bufferIndices, offsets);
}

void DrawCallMetaData<DrawCallMetaDataType::DGCSequence>::RecordCmds(
    vk::CommandBuffer cmd) const {
    sequenceBuffer->Execute(cmd);
}

void DrawCallMetaData<DrawCallMetaDataType::DGCPipelineInfo>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.setPolygonModeEXT(pipelineInfo.polygonMode);
    cmd.setCullModeEXT(pipelineInfo.cullMode);
    cmd.setRasterizationSamplesEXT(pipelineInfo.rasterSampleCount);
    cmd.setColorBlendEnableEXT(pipelineInfo.colorBlendInfo.firstAttachment,
                               pipelineInfo.colorBlendInfo.enableColorBlend);
    cmd.setColorBlendEquationEXT(pipelineInfo.colorBlendInfo.firstAttachment,
                                 pipelineInfo.colorBlendInfo.equations);
    cmd.setDepthTestEnableEXT(pipelineInfo.enableDepthTest);
    cmd.setDepthWriteEnableEXT(pipelineInfo.enableDepthWrite);
    cmd.setDepthCompareOpEXT(pipelineInfo.depthCompareOp);
}

void DrawCallMetaData<DrawCallMetaDataType::Copy>::RecordCmds(
    vk::CommandBuffer cmd) const {
    if (auto buffer2Buffer = ::std::get_if<vk::CopyBufferInfo2>(&info)) {
        cmd.copyBuffer2(buffer2Buffer);
    } else if (auto buffer2Image =
                   ::std::get_if<vk::CopyBufferToImageInfo2>(&info)) {
        cmd.copyBufferToImage2(buffer2Image);
    } else if (auto image2Image = ::std::get_if<vk::CopyImageInfo2>(&info)) {
        cmd.copyImage2(image2Image);
    } else if (auto image2Buffer =
                   ::std::get_if<vk::CopyImageToBufferInfo2>(&info)) {
        cmd.copyImageToBuffer2(image2Buffer);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core