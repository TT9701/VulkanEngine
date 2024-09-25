#include "DrawCallManager.h"
#include <ranges>

namespace IntelliDesign_NS::Vulkan::Core {

void DrawCallManager::AddArgument_ClearColorImage(
    vk::Image image, vk::ImageLayout layout,
    vk::ClearColorValue const& clearValue,
    std::initializer_list<vk::ImageSubresourceRange> const& ranges) {
    DrawCallMetaData<DrawCallMetaDataType::ClearColorImage> metaData;
    metaData.image = image;
    metaData.layout = layout;
    metaData.clearValue = clearValue;
    metaData.ranges = ranges;

    PushMetaDataMapping(DrawCallMetaDataType::ClearColorImage);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_ClearDepthStencilImage(
    vk::Image image, vk::ImageLayout layout,
    vk::ClearDepthStencilValue const& clearValue,
    std::initializer_list<vk::ImageSubresourceRange> const& ranges) {
    DrawCallMetaData<DrawCallMetaDataType::ClearDepthStencilImage> metaData;
    metaData.image = image;
    metaData.layout = layout;
    metaData.clearValue = clearValue;
    metaData.ranges = ranges;

    PushMetaDataMapping(DrawCallMetaDataType::ClearDepthStencilImage);
    mMetaDatas.emplace_back(metaData);
}

namespace {

void PushBarrierMapping(
    DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>& metaData,
    Type_STLUnorderedMap_String<uint32_t>& mapping,
    Type_STLVector<Type_STLString> const& names,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
    Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
    Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers) {
    uint32_t index = 0;
    if (!imgBarriers.empty()) {
        metaData.imgBarriers =
            ::std::make_optional<Type_STLVector<vk::ImageMemoryBarrier2>>();
        metaData.imgBarriers->reserve(imgBarriers.size());
        for (auto const& b : imgBarriers) {
            metaData.imgBarriers->emplace_back(b);
            mapping.emplace(names[index], index);
            ++index;
        }
    }

    if (!memBarriers.empty()) {
        metaData.memBarriers =
            ::std::make_optional<Type_STLVector<vk::MemoryBarrier2>>();
        metaData.memBarriers->reserve(memBarriers.size());
        for (auto const& b : memBarriers) {
            metaData.memBarriers->emplace_back(b);
            mapping.emplace(names[index], index);
            ++index;
        }
    }

    if (!bufBarriers.empty()) {
        metaData.bufBarriers =
            ::std::make_optional<Type_STLVector<vk::BufferMemoryBarrier2>>();
        metaData.bufBarriers->reserve(bufBarriers.size());
        for (auto const& b : bufBarriers) {
            metaData.bufBarriers->emplace_back(b);
            mapping.emplace(names[index], index);
            ++index;
        }
    }
}

}  // namespace

void DrawCallManager::AddArgument_MemoryBarriers_BeforePass(
    ::std::initializer_list<Type_STLString> const& names,
    std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers,
    std::initializer_list<vk::MemoryBarrier2> const& memBarriers,
    std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers) {
    assert(names.size()
           == imgBarriers.size() + memBarriers.size() + bufBarriers.size());

    mBarriers_BeforePass = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>();
    auto& metaData = mBarriers_BeforePass.value();

    PushBarrierMapping(metaData, mMapping.beforePass, names, imgBarriers,
                       memBarriers, bufBarriers);
}

void DrawCallManager::AddArgument_MemoryBarriers_AfterPass(
    ::std::initializer_list<Type_STLString> const& names,
    std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers,
    std::initializer_list<vk::MemoryBarrier2> const& memBarriers,
    std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers) {
    assert(names.size()
           == imgBarriers.size() + memBarriers.size() + bufBarriers.size());

    mBarriers_AfterPass = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>();
    auto& metaData = mBarriers_AfterPass.value();

    PushBarrierMapping(metaData, mMapping.afterPass, names, imgBarriers,
                       memBarriers, bufBarriers);
}

void DrawCallManager::AddArgument_RenderingInfo(
    vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
    std::initializer_list<vk::RenderingAttachmentInfo> const& colorAttachments,
    vk::RenderingAttachmentInfo const& depthStencilAttachment,
    vk::RenderingFlags flags) {
    mRenderingInfo = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>();
    auto& metaData = mRenderingInfo.value();
    bool bHasDepth = depthStencilAttachment.imageView != VK_NULL_HANDLE;

    metaData.colorAttachments = colorAttachments;
    if (bHasDepth)
        metaData.depthStencilAttachment = depthStencilAttachment;

    metaData.info.setFlags(flags)
        .setRenderArea(renderArea)
        .setLayerCount(layerCount)
        .setViewMask(viewMask)
        .setColorAttachments(metaData.colorAttachments);
    if (bHasDepth) {
        metaData.info.setPDepthAttachment(
            &metaData.depthStencilAttachment.value());
    }
}

void DrawCallManager::AddArgument_Viewport(
    uint32_t firstViewport,
    ::std::initializer_list<vk::Viewport> const& viewports) {
    DrawCallMetaData<DrawCallMetaDataType::Viewport> metaData;
    metaData.firstViewport = firstViewport;
    metaData.viewports = viewports;

    PushMetaDataMapping(DrawCallMetaDataType::Viewport);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Scissor(
    uint32_t firstScissor,
    ::std::initializer_list<vk::Rect2D> const& scissors) {
    DrawCallMetaData<DrawCallMetaDataType::Scissor> metaData;
    metaData.firstScissor = firstScissor;
    metaData.scissors = scissors;

    PushMetaDataMapping(DrawCallMetaDataType::Scissor);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                           vk::Pipeline pipeline) {
    DrawCallMetaData<DrawCallMetaDataType::Pipeline> metaData;
    metaData.bindPoint = bindPoint;
    metaData.pipeline = pipeline;

    PushMetaDataMapping(DrawCallMetaDataType::Pipeline);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_PushConstant(vk::PipelineLayout layout,
                                               vk::ShaderStageFlags stage,
                                               uint32_t offset, uint32_t size,
                                               const void* pValues) {
    DrawCallMetaData<DrawCallMetaDataType::PushContant> metaData;
    metaData.layout = layout;
    metaData.stage = stage;
    metaData.offset = offset;
    metaData.size = size;
    metaData.pValues = pValues;

    PushMetaDataMapping(DrawCallMetaDataType::PushContant);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DescriptorBuffer(
    std::initializer_list<vk::DeviceAddress> const& addresses) {
    DrawCallMetaData<DrawCallMetaDataType::DescriptorBuffer> metaData;
    metaData.addresses = addresses;

    PushMetaDataMapping(DrawCallMetaDataType::DescriptorBuffer);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DescriptorSet(
    vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
    uint32_t firstSet, std::initializer_list<uint32_t> const& bufferIndices,
    std::initializer_list<vk::DeviceSize> const& offsets) {
    DrawCallMetaData<DrawCallMetaDataType::DescriptorSet> metaData;
    metaData.bindPoint = bindPoint;
    metaData.layout = layout;
    metaData.firstSet = firstSet;
    metaData.bufferIndices = bufferIndices;
    metaData.offsets = offsets;

    PushMetaDataMapping(DrawCallMetaDataType::DescriptorSet);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_IndexBuffer(vk::Buffer buffer,
                                              vk::DeviceSize offset,
                                              vk::IndexType type) {
    DrawCallMetaData<DrawCallMetaDataType::IndexBuffer> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.type = type;

    PushMetaDataMapping(DrawCallMetaDataType::IndexBuffer);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                                     vk::DeviceSize offset,
                                                     uint32_t drawCount,
                                                     uint32_t stride) {
    DrawCallMetaData<DrawCallMetaDataType::DrawIndexedIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.drawCount = drawCount;
    metaData.stride = stride;

    PushMetaDataMapping(DrawCallMetaDataType::DrawIndexedIndirect);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Draw(uint32_t vertexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstVertex,
                                       uint32_t firstInstance) {
    DrawCallMetaData<DrawCallMetaDataType::Draw> metaData;
    metaData.vertexCount = vertexCount;
    metaData.instanceCount = instanceCount;
    metaData.firstVertex = firstVertex;
    metaData.firstInstance = firstInstance;

    PushMetaDataMapping(DrawCallMetaDataType::Draw);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DispatchIndirect(vk::Buffer buffer,
                                                   vk::DeviceSize offset) {
    DrawCallMetaData<DrawCallMetaDataType::DispatchIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;

    PushMetaDataMapping(DrawCallMetaDataType::DispatchIndirect);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Dispatch(uint32_t x, uint32_t y, uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::Dispatch> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

    PushMetaDataMapping(DrawCallMetaDataType::Dispatch);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DrawMeshTasksIndirect(vk::Buffer buffer,
                                                        vk::DeviceSize offset,
                                                        uint32_t drawCount,
                                                        uint32_t stride) {
    DrawCallMetaData<DrawCallMetaDataType::DrawMeshTasksIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.drawCount = drawCount;
    metaData.stride = stride;

    PushMetaDataMapping(DrawCallMetaDataType::DrawMeshTasksIndirect);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DrawMeshTask(uint32_t x, uint32_t y,
                                               uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::DrawMeshTask> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

    PushMetaDataMapping(DrawCallMetaDataType::DrawMeshTask);
    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::UpdateArgument_RenderArea(vk::Rect2D renderArea) {
    mRenderingInfo.value().info.setRenderArea(renderArea);
}

void DrawCallManager::UpdateArgument_Viewport(
    uint32_t firstViewport,
    std::initializer_list<vk::Viewport> const& viewports, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::Viewport>(index);
    data.firstViewport = firstViewport;
    data.viewports = viewports;
}

void DrawCallManager::UpdateArgument_Scissor(
    uint32_t firstScissor, std::initializer_list<vk::Rect2D> const& scissors,
    uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::Scissor>(index);
    data.firstScissor = firstScissor;
    data.scissors = scissors;
}

void DrawCallManager::UpdateArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                              vk::Pipeline pipeline,
                                              uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::Pipeline>(index);
    data.bindPoint = bindPoint;
    data.pipeline = pipeline;
}

void DrawCallManager::UpdateArgument_PushConstant(
    vk::PipelineLayout layout, vk::ShaderStageFlags stage, uint32_t offset,
    uint32_t size, const void* pValues, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::PushContant>(index);
    data.layout = layout;
    data.stage = stage;
    data.offset = offset;
    data.size = size;
    data.pValues = pValues;
}

void DrawCallManager::UpdateArgument_DescriptorBuffer(
    std::initializer_list<vk::DeviceAddress> const& addresses, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::DescriptorBuffer>(index);
    data.addresses = addresses;
}

void DrawCallManager::UpdateArgument_DescriptorSet(
    vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
    uint32_t firstSet, std::initializer_list<uint32_t> const& bufferIndices,
    std::initializer_list<vk::DeviceSize> const& offsets, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::DescriptorSet>(index);
    data.bindPoint = bindPoint;
    data.layout = layout;
    data.firstSet = firstSet;
    data.bufferIndices = bufferIndices;
    data.offsets = offsets;
}

void DrawCallManager::UpdateArgument_IndexBuffer(vk::Buffer buffer,
                                                 vk::DeviceSize offset,
                                                 vk::IndexType type,
                                                 uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::IndexBuffer>(index);
    data.buffer = buffer;
    data.offset = offset;
    data.type = type;
}

void DrawCallManager::UpdateArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                                        vk::DeviceSize offset,
                                                        uint32_t drawCount,
                                                        uint32_t stride,
                                                        uint32_t index) {
    auto& data =
        FindMetaDataRef<DrawCallMetaDataType::DrawIndexedIndirect>(index);
    data.buffer = buffer;
    data.offset = offset;
    data.drawCount = drawCount;
    data.stride = stride;
}

void DrawCallManager::UpdateArgument_Draw(uint32_t vertexCount,
                                          uint32_t instanceCount,
                                          uint32_t firstVertex,
                                          uint32_t firstInstance,
                                          uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::Draw>(index);
    data.vertexCount = vertexCount;
    data.instanceCount = instanceCount;
    data.firstVertex = firstVertex;
    data.firstInstance = firstInstance;
}

void DrawCallManager::UpdateArgument_DispatchIndirect(vk::Buffer buffer,
                                                      vk::DeviceSize offset,
                                                      uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::DispatchIndirect>(index);
    data.buffer = buffer;
    data.offset = offset;
}

void DrawCallManager::UpdateArgument_Dispatch(uint32_t x, uint32_t y,
                                              uint32_t z, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::Dispatch>(index);
    data.x = x;
    data.y = y;
    data.z = z;
}

void DrawCallManager::UpdateArgument_DrawMeshTasksIndirect(
    vk::Buffer buffer, vk::DeviceSize offset, uint32_t drawCount,
    uint32_t stride, uint32_t index) {
    auto& data =
        FindMetaDataRef<DrawCallMetaDataType::DrawMeshTasksIndirect>(index);
    data.buffer = buffer;
    data.offset = offset;
    data.drawCount = drawCount;
    data.stride = stride;
}

void DrawCallManager::UpdateArgument_DrawMeshTask(uint32_t x, uint32_t y,
                                                  uint32_t z, uint32_t index) {
    auto& data = FindMetaDataRef<DrawCallMetaDataType::DrawMeshTask>(index);
    data.x = x;
    data.y = y;
    data.z = z;
}

void DrawCallManager::UpdateArgument_Attachments(
    Type_STLVector<int> const& indices,
    Type_STLVector<vk::RenderingAttachmentInfo> const& attachments) {
    for (uint32_t i = 0; i < indices.size(); ++i) {
        if (indices[i] == -1)
            mRenderingInfo.value().depthStencilAttachment = attachments[i];
        else {
            mRenderingInfo.value().colorAttachments[indices[i]] =
                attachments[i];
        }
    }
}

void DrawCallManager::UpdateArgument_ImageBarriers_BeforePass(
    Type_STLVector<Type_STLString> const& names,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
    Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
    Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers) {
    auto imgSize = imgBarriers.size();
    auto memSize = memBarriers.size();
    auto bufSize = bufBarriers.size();
    assert(names.size() == imgSize + memSize + bufSize);

    uint32_t index {0};
    for (auto const& imgBar : imgBarriers) {
        auto mapIndex = mMapping.beforePass.at(names[index]);
        auto& old = mBarriers_BeforePass.value().imgBarriers.value()[mapIndex];
        old = imgBar;
        ++index;
    }

    for (auto const& memBar : memBarriers) {
        auto mapIndex = mMapping.beforePass.at(names[index]) - imgSize;
        auto& old = mBarriers_BeforePass.value().memBarriers.value()[mapIndex];
        old = memBar;
        ++index;
    }

    for (auto const& bufBar : bufBarriers) {
        auto mapIndex =
            mMapping.beforePass.at(names[index]) - imgSize - memSize;
        auto& old = mBarriers_BeforePass.value().bufBarriers.value()[mapIndex];
        old = bufBar;
        ++index;
    }
}

void DrawCallManager::UpdateArgument_ImageBarriers_AfterPass(
    Type_STLVector<Type_STLString> const& names,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
    Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
    Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers) {
    auto imgSize = imgBarriers.size();
    auto memSize = memBarriers.size();
    auto bufSize = bufBarriers.size();
    assert(names.size() == imgSize + memSize + bufSize);

    uint32_t index {0};
    for (auto const& imgBar : imgBarriers) {
        auto mapIndex = mMapping.afterPass.at(names[index]);
        auto& old = mBarriers_AfterPass.value().imgBarriers.value()[mapIndex];
        old = imgBar;
        ++index;
    }

    for (auto const& memBar : memBarriers) {
        auto mapIndex = mMapping.afterPass.at(names[index]) - imgSize;
        auto& old = mBarriers_AfterPass.value().memBarriers.value()[mapIndex];
        old = memBar;
        ++index;
    }

    for (auto const& bufBar : bufBarriers) {
        auto mapIndex = mMapping.afterPass.at(names[index]) - imgSize - memSize;
        auto& old = mBarriers_AfterPass.value().bufBarriers.value()[mapIndex];
        old = bufBar;
        ++index;
    }
}

void DrawCallManager::RecordCmd(vk::CommandBuffer cmd) const {
    if (mBarriers_BeforePass.has_value())
        mBarriers_BeforePass->RecordCmds(cmd);

    bool bIsGraphics {mRenderingInfo.has_value()};

    if (bIsGraphics)
        mRenderingInfo->RecordCmds(cmd);

    for (auto const& data : mMetaDatas)
        data->RecordCmds(cmd);

    if (bIsGraphics)
        cmd.endRendering();

    if (mBarriers_AfterPass.has_value())
        mBarriers_AfterPass->RecordCmds(cmd);
}

void DrawCallManager::Clear() {
    mBarriers_BeforePass.reset();
    mRenderingInfo.reset();
    mMetaDatas.clear();
    mBarriers_AfterPass.reset();
    mMapping = {};
}

void DrawCallManager::PushMetaDataMapping(DrawCallMetaDataType type) {
    if (mMapping.metaData.contains(type)) {
        mMapping.metaData.at(type).push_back(mMetaDatas.size());
    } else {
        mMapping.metaData.emplace(
            type, Type_STLVector<uint32_t> {
                      static_cast<uint32_t>(mMetaDatas.size())});
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core