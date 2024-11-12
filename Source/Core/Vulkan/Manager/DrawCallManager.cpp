#include "DrawCallManager.h"

#include "Core/Vulkan/Native/Swapchain.h"
#include "RenderResourceManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

DrawCallManager::DrawCallManager(RenderResourceManager* manager, Swapchain* sc)
    : pRenderResManager(manager), pSwapchain(sc) {}

void DrawCallManager::AddArgument_ClearColorImage(
    const char* imageName, vk::ImageLayout layout,
    vk::ClearColorValue const& clearValue,
    Type_STLVector<vk::ImageSubresourceRange> const& ranges) {
    auto image = (*pRenderResManager)[imageName]->GetTexHandle();

    DrawCallMetaData<DrawCallMetaDataType::ClearColorImage> metaData;
    metaData.image = image;
    metaData.layout = layout;
    metaData.clearValue = clearValue;
    metaData.ranges = ranges;

    mMetaDatas.emplace_back(metaData);
    PushResourceMetaDataMapping(imageName, mMetaDatas.size() - 1);
}

void DrawCallManager::AddArgument_ClearDepthStencilImage(
    const char* imageName, vk::ImageLayout layout,
    vk::ClearDepthStencilValue const& clearValue,
    Type_STLVector<vk::ImageSubresourceRange> const& ranges) {
    auto image = (*pRenderResManager)[imageName]->GetTexHandle();

    DrawCallMetaData<DrawCallMetaDataType::ClearDepthStencilImage> metaData;
    metaData.image = image;
    metaData.layout = layout;
    metaData.clearValue = clearValue;
    metaData.ranges = ranges;

    mMetaDatas.emplace_back(metaData);
    PushResourceMetaDataMapping(imageName, mMetaDatas.size() - 1);
}

namespace {

void PushBarrierMapping(
    DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>& metaData,
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
            metaData.mapping.emplace(names[index],
                                     &metaData.imgBarriers->back());
            ++index;
        }
    }

    if (!memBarriers.empty()) {
        metaData.memBarriers =
            ::std::make_optional<Type_STLVector<vk::MemoryBarrier2>>();
        metaData.memBarriers->reserve(memBarriers.size());
        for (auto const& b : memBarriers) {
            metaData.memBarriers->emplace_back(b);
            metaData.mapping.emplace(names[index],
                                     &metaData.memBarriers->back());
            ++index;
        }
    }

    if (!bufBarriers.empty()) {
        metaData.bufBarriers =
            ::std::make_optional<Type_STLVector<vk::BufferMemoryBarrier2>>();
        metaData.bufBarriers->reserve(bufBarriers.size());
        for (auto const& b : bufBarriers) {
            metaData.bufBarriers->emplace_back(b);
            metaData.mapping.emplace(names[index],
                                     &metaData.bufBarriers->back());
            ++index;
        }
    }
}

}  // namespace

void DrawCallManager::AddArgument_Barriers(
    Type_STLVector<Type_STLString> const& names,
    Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
    Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
    Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers) {
    assert(names.size()
           == imgBarriers.size() + memBarriers.size() + bufBarriers.size());

    mBarriers = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>();
    auto& metaData = mBarriers.value();

    PushBarrierMapping(metaData, names, imgBarriers, memBarriers, bufBarriers);
}

void DrawCallManager::AddArgument_RenderingInfo(
    vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
    Type_STLVector<RenderingAttachmentInfo> const& colorAttachments,
    RenderingAttachmentInfo const& depthStencilAttachment,
    vk::RenderingFlags flags) {
    mRenderingInfo = ::std::make_optional<
        DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>();
    auto& metaData = mRenderingInfo.value();
    bool bHasDepth = depthStencilAttachment.info.imageView != VK_NULL_HANDLE;

    Type_STLVector<vk::RenderingAttachmentInfo> colors(colorAttachments.size());
    uint32_t count {0};
    for (auto const& attachment : colorAttachments) {
        colors[count] = attachment.info;
        if (attachment.imageName != Type_STLString {"_Swapchain_"})
            colors[count].imageView =
                (*pRenderResManager)[attachment.imageName.c_str()]
                    ->GetTexViewHandle(attachment.viewName.c_str());

        Type_STLString mappingName {attachment.imageName};
        mappingName.append("@").append(attachment.viewName);
        metaData.mapping.emplace(mappingName, count);

        ++count;
    }
    metaData.colorAttachments = colors;

    if (bHasDepth) {
        metaData.depthStencilAttachment = depthStencilAttachment.info;
        metaData.depthStencilAttachment.value().imageView =
            (*pRenderResManager)[depthStencilAttachment.imageName.c_str()]
                ->GetTexViewHandle(depthStencilAttachment.viewName.c_str());
        Type_STLString mappingName {depthStencilAttachment.imageName};
        mappingName.append("@").append(depthStencilAttachment.viewName);
        metaData.mapping.emplace(mappingName, -1);
    }

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

void DrawCallManager::AddArgument_RenderingInfo(
    vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
    Type_STLVector<vk::RenderingAttachmentInfo> const& colorAttachments,
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
    uint32_t firstViewport, Type_STLVector<vk::Viewport> const& viewports) {
    DrawCallMetaData<DrawCallMetaDataType::Viewport> metaData;
    metaData.firstViewport = firstViewport;
    metaData.viewports = viewports;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Scissor(
    uint32_t firstScissor, Type_STLVector<vk::Rect2D> const& scissors) {
    DrawCallMetaData<DrawCallMetaDataType::Scissor> metaData;
    metaData.firstScissor = firstScissor;
    metaData.scissors = scissors;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                           vk::Pipeline pipeline) {
    DrawCallMetaData<DrawCallMetaDataType::Pipeline> metaData;
    metaData.bindPoint = bindPoint;
    metaData.pipeline = pipeline;

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

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DescriptorBuffer(
    Type_STLVector<vk::DeviceAddress> const& addresses) {
    DrawCallMetaData<DrawCallMetaDataType::DescriptorBuffer> metaData;
    metaData.addresses = addresses;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DescriptorSet(
    vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
    uint32_t firstSet, Type_STLVector<uint32_t> const& bufferIndices,
    Type_STLVector<vk::DeviceSize> const& offsets) {
    DrawCallMetaData<DrawCallMetaDataType::DescriptorSet> metaData;
    metaData.bindPoint = bindPoint;
    metaData.layout = layout;
    metaData.firstSet = firstSet;
    metaData.bufferIndices = bufferIndices;
    metaData.offsets = offsets;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_IndexBuffer(vk::Buffer buffer,
                                              vk::DeviceSize offset,
                                              vk::IndexType type) {
    DrawCallMetaData<DrawCallMetaDataType::IndexBuffer> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.type = type;

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

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DrawIndiret(vk::Buffer buffer,
                                              vk::DeviceSize offset,
                                              uint32_t drawCount,
                                              uint32_t stride) {
    DrawCallMetaData<DrawCallMetaDataType::DrawIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;
    metaData.drawCount = drawCount;
    metaData.stride = stride;

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

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DispatchIndirect(vk::Buffer buffer,
                                                   vk::DeviceSize offset) {
    DrawCallMetaData<DrawCallMetaDataType::DispatchIndirect> metaData;
    metaData.buffer = buffer;
    metaData.offset = offset;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_Dispatch(uint32_t x, uint32_t y, uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::Dispatch> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

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

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_DrawMeshTask(uint32_t x, uint32_t y,
                                               uint32_t z) {
    DrawCallMetaData<DrawCallMetaDataType::DrawMeshTask> metaData;
    metaData.x = x;
    metaData.y = y;
    metaData.z = z;

    mMetaDatas.emplace_back(metaData);
}

void DrawCallManager::AddArgument_CopyBufferToBuffer(const char* src,
                                                     const char* dst,
                                                     vk::BufferCopy2 region) {
    mMetaDatas.emplace_back(DrawCallMetaData<DrawCallMetaDataType::Copy> {});
    auto& metaData = mMetaDatas.back().Get<DrawCallMetaDataType::Copy>();

    metaData.copyRegion.emplace<vk::BufferCopy2>(region);
    metaData.info.emplace<vk::CopyBufferInfo2>();

    auto pRegion = ::std::get_if<vk::BufferCopy2>(&metaData.copyRegion);
    auto& info = ::std::get<vk::CopyBufferInfo2>(metaData.info);
    info.setSrcBuffer((*pRenderResManager)[src]->GetBufferHandle())
        .setDstBuffer((*pRenderResManager)[dst]->GetBufferHandle())
        .setRegionCount(1)
        .setPRegions(pRegion);
}

void DrawCallManager::AddArgument_CopyBufferToImage(
    const char* src, const char* dst, vk::BufferImageCopy2 region) {
    mMetaDatas.emplace_back(DrawCallMetaData<DrawCallMetaDataType::Copy> {});
    auto& metaData = mMetaDatas.back().Get<DrawCallMetaDataType::Copy>();

    metaData.copyRegion.emplace<vk::BufferImageCopy2>(region);
    metaData.info.emplace<vk::CopyBufferToImageInfo2>();

    auto pRegion = ::std::get_if<vk::BufferImageCopy2>(&metaData.copyRegion);

    auto& info = ::std::get<vk::CopyBufferToImageInfo2>(metaData.info);
    info.setSrcBuffer((*pRenderResManager)[src]->GetBufferHandle())
        .setDstImage((*pRenderResManager)[dst]->GetTexHandle())
        .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
        .setRegionCount(1)
        .setPRegions(pRegion);
}

void DrawCallManager::AddArgument_CopyImageToBuffer(
    const char* src, const char* dst, vk::BufferImageCopy2 region) {
    mMetaDatas.emplace_back(DrawCallMetaData<DrawCallMetaDataType::Copy> {});
    auto& metaData = mMetaDatas.back().Get<DrawCallMetaDataType::Copy>();

    metaData.copyRegion.emplace<vk::BufferImageCopy2>(region);
    metaData.info.emplace<vk::CopyImageToBufferInfo2>();

    auto pRegion = ::std::get_if<vk::BufferImageCopy2>(&metaData.copyRegion);

    auto& info = ::std::get<vk::CopyImageToBufferInfo2>(metaData.info);
    info.setSrcImage((*pRenderResManager)[src]->GetTexHandle())
        .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setDstBuffer((*pRenderResManager)[dst]->GetBufferHandle())
        .setRegionCount(1)
        .setPRegions(pRegion);
}

void DrawCallManager::AddArgument_CopyImageToImage(const char* src,
                                                   const char* dst,
                                                   vk::ImageCopy2 region) {
    mMetaDatas.emplace_back(DrawCallMetaData<DrawCallMetaDataType::Copy> {});
    auto& metaData = mMetaDatas.back().Get<DrawCallMetaDataType::Copy>();

    metaData.copyRegion.emplace<vk::ImageCopy2>(region);
    metaData.info.emplace<vk::CopyImageToBufferInfo2>();

    auto pRegion = ::std::get_if<vk::ImageCopy2>(&metaData.copyRegion);

    auto& info = ::std::get<vk::CopyImageInfo2>(metaData.info);
    info.setSrcImage((*pRenderResManager)[src]->GetTexHandle())
        .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setDstImage((*pRenderResManager)[dst]->GetTexHandle())
        .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
        .setRegionCount(1)
        .setPRegions(pRegion);
}

void DrawCallManager::UpdateArgument_RenderArea(vk::Rect2D renderArea) {
    mRenderingInfo.value().info.setRenderArea(renderArea);
}

void DrawCallManager::UpdateArgument_Viewport(
    uint32_t firstViewport, Type_STLVector<vk::Viewport> const& viewports,
    uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::Viewport>(index);
    data->firstViewport = firstViewport;
    data->viewports = viewports;
}

void DrawCallManager::UpdateArgument_Scissor(
    uint32_t firstScissor, Type_STLVector<vk::Rect2D> const& scissors,
    uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::Scissor>(index);
    data->firstScissor = firstScissor;
    data->scissors = scissors;
}

void DrawCallManager::UpdateArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                              vk::Pipeline pipeline,
                                              uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::Pipeline>(index);
    data->bindPoint = bindPoint;
    data->pipeline = pipeline;
}

void DrawCallManager::UpdateArgument_PushConstant(
    vk::PipelineLayout layout, vk::ShaderStageFlags stage, uint32_t offset,
    uint32_t size, const void* pValues, uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::PushContant>(index);
    data->layout = layout;
    data->stage = stage;
    data->offset = offset;
    data->size = size;
    data->pValues = pValues;
}

void DrawCallManager::UpdateArgument_IndexBuffer(vk::Buffer buffer,
                                                 vk::DeviceSize offset,
                                                 vk::IndexType type,
                                                 uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::IndexBuffer>(index);
    data->buffer = buffer;
    data->offset = offset;
    data->type = type;
}

void DrawCallManager::UpdateArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                                        vk::DeviceSize offset,
                                                        uint32_t drawCount,
                                                        uint32_t stride,
                                                        uint32_t index) {
    auto data =
        FindMetaDataPtr<DrawCallMetaDataType::DrawIndexedIndirect>(index);
    data->buffer = buffer;
    data->offset = offset;
    data->drawCount = drawCount;
    data->stride = stride;
}

void DrawCallManager::UpdateArgument_Draw(uint32_t vertexCount,
                                          uint32_t instanceCount,
                                          uint32_t firstVertex,
                                          uint32_t firstInstance,
                                          uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::Draw>(index);
    data->vertexCount = vertexCount;
    data->instanceCount = instanceCount;
    data->firstVertex = firstVertex;
    data->firstInstance = firstInstance;
}

void DrawCallManager::UpdateArgument_DispatchIndirect(vk::Buffer buffer,
                                                      vk::DeviceSize offset,
                                                      uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::DispatchIndirect>(index);
    data->buffer = buffer;
    data->offset = offset;
}

void DrawCallManager::UpdateArgument_Dispatch(uint32_t x, uint32_t y,
                                              uint32_t z, uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::Dispatch>(index);
    data->x = x;
    data->y = y;
    data->z = z;
}

void DrawCallManager::UpdateArgument_DrawMeshTasksIndirect(
    vk::Buffer buffer, vk::DeviceSize offset, uint32_t drawCount,
    uint32_t stride, uint32_t index) {
    auto data =
        FindMetaDataPtr<DrawCallMetaDataType::DrawMeshTasksIndirect>(index);
    data->buffer = buffer;
    data->offset = offset;
    data->drawCount = drawCount;
    data->stride = stride;
}

void DrawCallManager::UpdateArgument_DrawMeshTask(uint32_t x, uint32_t y,
                                                  uint32_t z, uint32_t index) {
    auto data = FindMetaDataPtr<DrawCallMetaDataType::DrawMeshTask>(index);
    data->x = x;
    data->y = y;
    data->z = z;
}

void DrawCallManager::UpdateArgument_Attachments(
    Type_STLVector<Type_STLString> const& names) {
    for (auto const& name : names) {
        auto offset = name.find('@');
        auto imageName = name.substr(0, offset);
        auto viewName = name.substr(offset + 1, name.size() - offset);

        auto index = mRenderingInfo->mapping.at(name);
        if (index == -1) {
            mRenderingInfo->depthStencilAttachment->imageView =
                (*pRenderResManager)[imageName.c_str()]->GetTexViewHandle(
                    viewName.c_str());
        } else {
            if (imageName == "_Swapchain_") {
                mRenderingInfo->colorAttachments[index].imageView =
                    pSwapchain->GetImageViewHandle(
                        pSwapchain->GetCurrentImageIndex());
            } else {
                mRenderingInfo->colorAttachments[index].imageView =
                    (*pRenderResManager)[imageName.c_str()]->GetTexViewHandle(
                        viewName.c_str());
            }
        }
    }
}

void DrawCallManager::UpdateArgument_Attachments(
    Type_STLVector<int> const& indices,
    Type_STLVector<vk::RenderingAttachmentInfo> const& attachments) {
    for (uint32_t i = 0; i < indices.size(); ++i) {
        if (indices[i] == -1)
            mRenderingInfo->depthStencilAttachment = attachments[i];
        else {
            mRenderingInfo->colorAttachments[indices[i]] = attachments[i];
        }
    }
}

void DrawCallManager::UpdateArgument_Barriers(
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
        auto ptr = std::get<vk::ImageMemoryBarrier2*>(
            mBarriers->mapping.at(names[index]));
        *ptr = imgBar;
        ++index;
    }

    for (auto const& memBar : memBarriers) {
        auto ptr =
            std::get<vk::MemoryBarrier2*>(mBarriers->mapping.at(names[index]));
        *ptr = memBar;
        ++index;
    }

    for (auto const& bufBar : bufBarriers) {
        auto ptr = std::get<vk::BufferMemoryBarrier2*>(
            mBarriers->mapping.at(names[index]));
        *ptr = bufBar;
        ++index;
    }
}

void DrawCallManager::UpdateArgument_Barriers(
    Type_STLVector<Type_STLString> const& names) {
    for (auto const& name : names) {
        auto var = mBarriers->mapping.at(name);
        if (auto pib = ::std::get_if<vk::ImageMemoryBarrier2*>(&var)) {
            if (name == "_Swapchain_") {
                (*pib)->setImage(pSwapchain->GetCurrentImage().GetTexHandle());
            } else {
                (*pib)->setImage(
                    (*pRenderResManager)[name.c_str()]->GetTexHandle());
            }
        } else if (auto pbb = ::std::get_if<vk::BufferMemoryBarrier2*>(&var)) {
            (*pbb)->setBuffer(
                (*pRenderResManager)[name.c_str()]->GetBufferHandle());
        } else {
            throw ::std::runtime_error(
                "resource is not needed for vk::MemoryBarrier2");
        }
    }
}

void DrawCallManager::UpdateArgument_CopySrc(const char* name, uint32_t index) {
    auto metaData = FindMetaDataPtr<DrawCallMetaDataType::Copy>(index);

    if (auto b2bInfo = ::std::get_if<vk::CopyBufferInfo2>(&metaData->info)) {
        b2bInfo->setSrcBuffer((*pRenderResManager)[name]->GetBufferHandle());
    } else if (auto b2iInfo =
                   ::std::get_if<vk::CopyBufferToImageInfo2>(&metaData->info)) {
        b2iInfo->setSrcBuffer((*pRenderResManager)[name]->GetBufferHandle());
    } else if (auto i2iInfo =
                   ::std::get_if<vk::CopyImageInfo2>(&metaData->info)) {
        i2iInfo->setSrcImage((*pRenderResManager)[name]->GetTexHandle());
    } else if (auto i2bInfo =
                   ::std::get_if<vk::CopyImageToBufferInfo2>(&metaData->info)) {
        i2bInfo->setSrcImage((*pRenderResManager)[name]->GetTexHandle());
    }
}

void DrawCallManager::UpdateArgument_CopyDst(const char* name, uint32_t index) {
    auto metaData = FindMetaDataPtr<DrawCallMetaDataType::Copy>(index);

    if (auto b2bInfo = ::std::get_if<vk::CopyBufferInfo2>(&metaData->info)) {
        b2bInfo->setDstBuffer((*pRenderResManager)[name]->GetBufferHandle());
    } else if (auto b2iInfo =
                   ::std::get_if<vk::CopyBufferToImageInfo2>(&metaData->info)) {
        b2iInfo->setDstImage((*pRenderResManager)[name]->GetTexHandle());
    } else if (auto i2iInfo =
                   ::std::get_if<vk::CopyImageInfo2>(&metaData->info)) {
        i2iInfo->setDstImage((*pRenderResManager)[name]->GetTexHandle());
    } else if (auto i2bInfo =
                   ::std::get_if<vk::CopyImageToBufferInfo2>(&metaData->info)) {
        i2bInfo->setDstBuffer((*pRenderResManager)[name]->GetBufferHandle());
    }
}

void DrawCallManager::UpdateArgument_OnResize(vk::Extent2D extent) {
    if (mRenderingInfo.has_value())
        mRenderingInfo->info.renderArea.extent = extent;

    if (auto viewport = FindMetaDataPtr<DrawCallMetaDataType::Viewport>())
        viewport->viewports.front()
            .setWidth(extent.width)
            .setHeight(extent.height);

    if (auto scissor = FindMetaDataPtr<DrawCallMetaDataType::Scissor>())
        scissor->scissors.front().setExtent(extent);

    Type_STLVector<Type_STLString> beforePassBarrierResourceNames {};
    Type_STLVector<Type_STLString> afterPassBarrierResourceNames {};
    Type_STLVector<Type_STLString> attachmentResourceNames {};

    for (auto const& name :
         pRenderResManager->GetResourceNames_SrcreenSizeRelated()) {
        if (mBarriers && mBarriers->mapping.contains(name)) {
            beforePassBarrierResourceNames.push_back(name);
        }

        if (mRenderingInfo) {
            auto& mapping = mRenderingInfo->mapping;
            Type_STLString viewName {};
            for (auto const& [k, _] : mapping) {
                if (k.find(name) != Type_STLString::npos) {
                    auto offset = k.find('@');
                    viewName = k.substr(offset + 1, k.size() - offset);
                    break;
                }
            }
            auto mappingName =
                Type_STLString {name}.append("@").append(viewName);

            if (mRenderingInfo->mapping.contains(mappingName)) {
                attachmentResourceNames.push_back(mappingName);
            }
        }

        if (mResourceMetaDataMapping.contains(name)) {
            auto& indices = mResourceMetaDataMapping.at(name);
            for (auto index : indices) {
                mMetaDatas[index]->UpdateRenderResource(pRenderResManager,
                                                        name);
            }
        }
    }

    UpdateArgument_Barriers(beforePassBarrierResourceNames);
    UpdateArgument_Attachments(attachmentResourceNames);
}

void DrawCallManager::RecordCmd(vk::CommandBuffer cmd) const {
    if (mBarriers)
        mBarriers->RecordCmds(cmd);

    bool bIsGraphics {mRenderingInfo};

    if (bIsGraphics)
        mRenderingInfo->RecordCmds(cmd);

    for (auto const& data : mMetaDatas)
        data->RecordCmds(cmd);

    if (bIsGraphics)
        cmd.endRendering();
}

void DrawCallManager::Clear() {
    mBarriers.reset();
    mRenderingInfo.reset();
    mMetaDatas.clear();
    mResourceMetaDataMapping.clear();
}

void DrawCallManager::PushResourceMetaDataMapping(const char* name,
                                                  uint32_t index) {
    if (mResourceMetaDataMapping.contains(name)) {
        mResourceMetaDataMapping.at(name).emplace_back(index);
    } else {
        mResourceMetaDataMapping.emplace(name,
                                         Type_STLVector<uint32_t> {index});
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core