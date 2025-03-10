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

void DrawCallMetaData<DrawCallMetaDataType::PushContant>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.pushConstants(layout, stage, offset, size, pValues);
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

void DrawCallMetaData<DrawCallMetaDataType::IndexBuffer>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.bindIndexBuffer(buffer, offset, type);
}

void DrawCallMetaData<DrawCallMetaDataType::DrawIndexedIndirect>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.drawIndexedIndirect(buffer, offset, drawCount, stride);
}

void DrawCallMetaData<DrawCallMetaDataType::DrawIndirect>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.drawIndirect(buffer, offset, drawCount, stride);
}

void DrawCallMetaData<DrawCallMetaDataType::Draw>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void DrawCallMetaData<DrawCallMetaDataType::DispatchIndirect>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.dispatchIndirect(buffer, offset);
}

void DrawCallMetaData<DrawCallMetaDataType::Dispatch>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.dispatch(x, y, z);
}

void DrawCallMetaData<DrawCallMetaDataType::DrawMeshTasksIndirect>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.drawMeshTasksIndirectEXT(buffer, offset, drawCount, stride);
}

void DrawCallMetaData<DrawCallMetaDataType::DrawMeshTask>::RecordCmds(
    vk::CommandBuffer cmd) const {
    cmd.drawMeshTasksEXT(x, y, z);
}

void DrawCallMetaData<DrawCallMetaDataType::DGCSequence>::RecordCmds(
    vk::CommandBuffer cmd) const {
    sequenceBuffer->Execute(cmd);
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