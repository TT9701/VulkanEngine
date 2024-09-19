#include "DrawCallManager.h"

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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::ClearColorImage,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::ClearDepthStencilImage,
                                  mMapping.count++);
}

namespace {
auto barriersInput =
    [](DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>& metaData,
       Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
       Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
       Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers) {
        if (!imgBarriers.empty()) {
            metaData.imgBarriers =
                ::std::make_optional<Type_STLVector<vk::ImageMemoryBarrier2>>();
            metaData.imgBarriers->reserve(imgBarriers.size());
            for (auto const& b : imgBarriers) {
                metaData.imgBarriers->emplace_back(b);
                metaData.imgBarriersMapping.mapping.emplace_back(
                    b.image, metaData.imgBarriersMapping.count++);
            }
        }

        if (!memBarriers.empty()) {
            metaData.memBarriers =
                ::std::make_optional<Type_STLVector<vk::MemoryBarrier2>>();
            metaData.memBarriers->reserve(memBarriers.size());
            for (auto const& b : memBarriers) {
                metaData.memBarriers->emplace_back(b);
                metaData.memBarriersMapping.mapping.emplace_back(
                    b.dstAccessMask, metaData.memBarriersMapping.count++);
            }
        }

        if (!bufBarriers.empty()) {
            metaData.bufBarriers = ::std::make_optional<
                Type_STLVector<vk::BufferMemoryBarrier2>>();
            metaData.bufBarriers->reserve(bufBarriers.size());
            for (auto const& b : bufBarriers) {
                metaData.bufBarriers->emplace_back(b);
                metaData.bufBarriersMapping.mapping.emplace_back(
                    b.buffer, metaData.bufBarriersMapping.count++);
            }
        }
    };
}  // namespace

void DrawCallManager::AddArgument_MemoryBarriers_BeforePass(
    std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers,
    std::initializer_list<vk::MemoryBarrier2> const& memBarriers,
    std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers) {
    mBarriers_BeforePass = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>();
    auto& metaData = mBarriers_BeforePass.value();

    barriersInput(metaData, imgBarriers, memBarriers, bufBarriers);
}

void DrawCallManager::AddArgument_MemoryBarriers_AfterPass(
    std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers,
    std::initializer_list<vk::MemoryBarrier2> const& memBarriers,
    std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers) {
    mBarriers_AfterPass = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>();
    auto& metaData = mBarriers_AfterPass.value();

    barriersInput(metaData, imgBarriers, memBarriers, bufBarriers);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::Viewport,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_Scissor(
    uint32_t firstScissor,
    ::std::initializer_list<vk::Rect2D> const& scissors) {
    DrawCallMetaData<DrawCallMetaDataType::Scissor> metaData;
    metaData.firstScissor = firstScissor;
    metaData.scissors = scissors;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::Scissor,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                           vk::Pipeline pipeline) {
    DrawCallMetaData<DrawCallMetaDataType::Pipeline> metaData;
    metaData.bindPoint = bindPoint;
    metaData.pipeline = pipeline;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::Pipeline,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::PushContant,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_DescriptorBuffer(
    std::initializer_list<vk::DeviceAddress> const& addresses) {
    DrawCallMetaData<DrawCallMetaDataType::DescriptorBuffer> metaData;
    metaData.addresses = addresses;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DescriptorBuffer,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DescriptorSet,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_IndexBuffer(vk::Buffer buffer,
                                              vk::DeviceSize offset,
                                              vk::IndexType type) {
    DrawCallMetaData<DrawCallMetaDataType::IndexBuffer> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.type = type;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::IndexBuffer,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DrawIndexedIndirect,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::Draw, mMapping.count++);
}

void DrawCallManager::AddArgument_DispatchIndirect(vk::Buffer buffer,
                                                   vk::DeviceSize offset) {
    DrawCallMetaData<DrawCallMetaDataType::DispatchIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DispatchIndirect,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_Dispatch(uint32_t x, uint32_t y, uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::Dispatch> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::Dispatch,
                                  mMapping.count++);
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

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DrawMeshTasksIndirect,
                                  mMapping.count++);
}

void DrawCallManager::AddArgument_DrawMeshTask(uint32_t x, uint32_t y,
                                               uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::DrawMeshTask> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

    mMetaDatas.emplace_back(metaData);
    mMapping.mapping.emplace_back(DrawCallMetaDataType::DrawMeshTask,
                                  mMapping.count++);
}

void DrawCallManager::UpdateArgument_ColorAttachments(
    std::initializer_list<vk::RenderingAttachmentInfo> const&
        colorAttachments) {
    mRenderingInfo.value().colorAttachments = colorAttachments;
}

void DrawCallManager::UpdateArgument_MemoryBarriers_BeforePass(
    Type_STLVector<vk::Image> const& indices,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers) {
    auto& barriers = mBarriers_BeforePass.value().imgBarriersMapping.mapping;
    for (uint32_t i = 0; i < indices.size(); ++i) {
        auto const& idx = indices[i];
        auto it =
            ::std::find_if(barriers.begin(), barriers.end(),
                           [&idx](::std::pair<vk::Image, uint32_t> mapping) {
                               return mapping.first == idx;
                           });
        mBarriers_BeforePass.value().imgBarriers.value()[it->second] =
            imgBarriers[i];
    }
}

void DrawCallManager::UpdateArgument_MemoryBarriers_AfterPass(
    Type_STLVector<vk::Image> const& indices,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers) {
    auto& barriers = mBarriers_AfterPass.value().imgBarriersMapping.mapping;
    for (uint32_t i = 0; i < indices.size(); ++i) {
        auto const& idx = indices[i];
        auto it =
            ::std::find_if(barriers.begin(), barriers.end(),
                           [&idx](::std::pair<vk::Image, uint32_t> mapping) {
                               return mapping.first == idx;
                           });
        mBarriers_AfterPass.value().imgBarriers.value()[it->second] =
            imgBarriers[i];
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

}  // namespace IntelliDesign_NS::Vulkan::Core