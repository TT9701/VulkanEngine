#pragma once

#include "Core/Vulkan/Native/DrawCallMetaData.h"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderResourceManager;
class Swapchain;

class DrawCallManager {
public:
    DrawCallManager(RenderResourceManager& manager);

    /*
     *   Add Argument methods
     */
    void AddArgument_ClearColorImage(
        const char* imageName, vk::ImageLayout layout,
        vk::ClearColorValue const& clearValue,
        Type_STLVector<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_ClearDepthStencilImage(
        const char* imageName, vk::ImageLayout layout,
        vk::ClearDepthStencilValue const& clearValue,
        Type_STLVector<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_ResetBuffer(const char* bufferName, vk::DeviceSize offset,
                                 vk::DeviceSize size);

    void AddArgument_Barriers(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers = {},
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers = {},
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers = {});

    void AddArgument_RenderingInfo(
        vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
        Type_STLVector<RenderingAttachmentInfo> const& colorAttachments,
        RenderingAttachmentInfo const& depthStencilAttachment = {},
        vk::RenderingFlags flags = {});

    void AddArgument_RenderingInfo(
        vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
        Type_STLVector<vk::RenderingAttachmentInfo> const& colorAttachments,
        vk::RenderingAttachmentInfo const& depthStencilAttachment = {},
        vk::RenderingFlags flags = {});

    void AddArgument_Viewport(uint32_t firstViewport,
                              Type_STLVector<vk::Viewport> const& viewports);

    void AddArgument_Scissor(uint32_t firstScissor,
                             Type_STLVector<vk::Rect2D> const& scissors);

    void AddArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                              vk::Pipeline pipeline);

    void AddArgument_DescriptorBuffer(
        Type_STLVector<vk::DeviceAddress> const& addresses);

    void AddArgument_DescriptorSet(
        vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
        uint32_t firstSet, Type_STLVector<uint32_t> const& bufferIndices,
        Type_STLVector<vk::DeviceSize> const& offsets);

    void AddArgument_DGCSequence(RenderResource const* sequenceBuffer);

    void AddArgument_DGCPipelineInfo(DGCPipelineInfo const& info);

    void AddArgument_CopyBufferToBuffer(const char* src, const char* dst,
                                        const vk::BufferCopy2* region);

    void AddArgument_CopyBufferToBuffer(
        const char* src, const char* dst,
        Type_STLVector<vk::BufferCopy2> const& regions);

    void AddArgument_CopyBufferToImage(
        const char* src, const char* dst,
        Type_STLVector<vk::BufferImageCopy2> const& regions);

    void AddArgument_CopyImageToBuffer(
        const char* src, const char* dst,
        Type_STLVector<vk::BufferImageCopy2> const& regions);

    void AddArgument_CopyImageToImage(
        const char* src, const char* dst,
        Type_STLVector<vk::ImageCopy2> const& regions);

    /*
     *   Update Argument methods
     */
    void UpdateArgument_RenderArea(vk::Rect2D renderArea);

    void UpdateArgument_Viewport(uint32_t firstViewport,
                                 Type_STLVector<vk::Viewport> const& viewports,
                                 uint32_t index = 0);

    void UpdateArgument_Scissor(uint32_t firstScissor,
                                Type_STLVector<vk::Rect2D> const& scissors,
                                uint32_t index = 0);

    void UpdateArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                 vk::Pipeline pipeline, uint32_t index = 0);

    void UpdateArgument_DescriptorBuffer(
        Type_STLVector<vk::DeviceAddress> const& addresses, uint32_t index = 0);

    void UpdateArgument_DescriptorSet(
        vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
        uint32_t firstSet, Type_STLVector<uint32_t> const& bufferIndices,
        Type_STLVector<vk::DeviceSize> const& offsets, uint32_t index = 0);

    // name = imageName@viewName
    void UpdateArgument_Attachments(
        Type_STLVector<Type_STLString> const& names);

    void UpdateArgument_Attachments(
        Type_STLVector<int> const& indices,
        Type_STLVector<vk::RenderingAttachmentInfo> const& attachments);

    void UpdateArgument_Barriers(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers);

    // only update resources
    void UpdateArgument_Barriers(Type_STLVector<Type_STLString> const& names);

    void UpdateArgument_DGCSequence(RenderResource const* buffer);
    void UpdateArgument_DGCPipeline(DGCPipelineInfo const& info);

    void UpdateArgument_CopySrc(const char* name, uint32_t index);
    void UpdateArgument_CopyDst(const char* name, uint32_t index);

    void UpdateArgument_OnResize(vk::Extent2D extent);

public:
    void RecordCmd(vk::CommandBuffer cmd) const;

    void Clear();

private:
    template <DrawCallMetaDataType Type>
    DrawCallMetaData<Type>* FindMetaDataPtr(uint32_t index = 0);

    void PushResourceMetaDataMapping(const char* name, uint32_t index);

private:
    RenderResourceManager& mRenderResManager;

    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers;
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>
        mRenderingInfo;
    Type_STLVector<Type_DrawCallMetaData_Unified> mMetaDatas {};

    Type_STLUnorderedMap_String<Type_STLVector<uint32_t>>
        mResourceMetaDataMapping;
};

template <DrawCallMetaDataType Type>
DrawCallMetaData<Type>* DrawCallManager::FindMetaDataPtr(uint32_t index) {
    Type_STLVector<Type_DrawCallMetaData_Unified*> temp {};
    auto it = mMetaDatas.begin();
    while (it != mMetaDatas.end()) {
        auto temp_it =
            ::std::find_if(it, mMetaDatas.end(),
                           [](Type_DrawCallMetaData_Unified const& data) {
                               return data.Get_Index() == Type;
                           });
        if (temp_it != mMetaDatas.end())
            temp.push_back(&*temp_it);
        else
            break;

        it = ++temp_it;
    }

    if (temp.empty())
        return nullptr;

    assert(index < temp.size());

    return &temp[index]->Get<Type>();
}

}  // namespace IntelliDesign_NS::Vulkan::Core