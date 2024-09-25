#pragma once

#include "Core/Vulkan/Native/DrawCallMetaData.h"

namespace IntelliDesign_NS::Vulkan::Core {

struct MetaDataResourceMapping {
    // "resource param" - barrier index
    Type_STLUnorderedMap_String<uint32_t> beforePass {};

    // DrawCallMetaDataType + index -> metaData
    Type_STLUnorderedMap<DrawCallMetaDataType, Type_STLVector<uint32_t>>
        metaData {};

    // "resource param" - barrier index
    Type_STLUnorderedMap_String<uint32_t> afterPass {};
};

class DrawCallManager {
public:
    /*
     *   Add Argument methods
     */
    void AddArgument_ClearColorImage(
        vk::Image image, vk::ImageLayout layout,
        vk::ClearColorValue const& clearValue,
        ::std::initializer_list<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_ClearDepthStencilImage(
        vk::Image image, vk::ImageLayout layout,
        vk::ClearDepthStencilValue const& clearValue,
        ::std::initializer_list<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_MemoryBarriers_BeforePass(
        ::std::initializer_list<Type_STLString> const& names,
        ::std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers =
            {},
        ::std::initializer_list<vk::MemoryBarrier2> const& memBarriers = {},
        ::std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers =
            {});

    void AddArgument_MemoryBarriers_AfterPass(
        ::std::initializer_list<Type_STLString> const& names,
        ::std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers =
            {},
        ::std::initializer_list<vk::MemoryBarrier2> const& memBarriers = {},
        ::std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers =
            {});

    void AddArgument_RenderingInfo(
        vk::Rect2D renderArea, uint32_t layerCount, uint32_t viewMask,
        ::std::initializer_list<vk::RenderingAttachmentInfo> const&
            colorAttachments,
        vk::RenderingAttachmentInfo const& depthStencilAttachment = {},
        vk::RenderingFlags flags = {});

    void AddArgument_Viewport(
        uint32_t firstViewport,
        ::std::initializer_list<vk::Viewport> const& viewports);

    void AddArgument_Scissor(
        uint32_t firstScissor,
        ::std::initializer_list<vk::Rect2D> const& scissors);

    void AddArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                              vk::Pipeline pipeline);

    void AddArgument_PushConstant(vk::PipelineLayout layout,
                                  vk::ShaderStageFlags stage, uint32_t offset,
                                  uint32_t size, const void* pValues);

    void AddArgument_DescriptorBuffer(
        ::std::initializer_list<vk::DeviceAddress> const& addresses);

    void AddArgument_DescriptorSet(
        vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
        uint32_t firstSet,
        ::std::initializer_list<uint32_t> const& bufferIndices,
        ::std::initializer_list<vk::DeviceSize> const& offsets);

    void AddArgument_IndexBuffer(vk::Buffer buffer, vk::DeviceSize offset,
                                 vk::IndexType type);

    void AddArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                        vk::DeviceSize offset,
                                        uint32_t drawCount, uint32_t stride);

    void AddArgument_Draw(uint32_t vertexCount, uint32_t instanceCount,
                          uint32_t firstVertex, uint32_t firstInstance);

    void AddArgument_DispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset);

    void AddArgument_Dispatch(uint32_t x, uint32_t y, uint32_t z);

    void AddArgument_DrawMeshTasksIndirect(vk::Buffer buffer,
                                           vk::DeviceSize offset,
                                           uint32_t drawCount, uint32_t stride);

    void AddArgument_DrawMeshTask(uint32_t x, uint32_t y, uint32_t z);

    /*
     *   Update Argument methods
     */
    void UpdateArgument_RenderArea(vk::Rect2D renderArea);

    void UpdateArgument_Viewport(
        uint32_t firstViewport,
        ::std::initializer_list<vk::Viewport> const& viewports,
        uint32_t index = 0);

    void UpdateArgument_Scissor(
        uint32_t firstScissor,
        ::std::initializer_list<vk::Rect2D> const& scissors,
        uint32_t index = 0);

    void UpdateArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                 vk::Pipeline pipeline, uint32_t index = 0);

    void UpdateArgument_PushConstant(vk::PipelineLayout layout,
                                     vk::ShaderStageFlags stage,
                                     uint32_t offset, uint32_t size,
                                     const void* pValues, uint32_t index = 0);

    void UpdateArgument_DescriptorBuffer(
        ::std::initializer_list<vk::DeviceAddress> const& addresses,
        uint32_t index = 0);

    void UpdateArgument_DescriptorSet(
        vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
        uint32_t firstSet,
        ::std::initializer_list<uint32_t> const& bufferIndices,
        ::std::initializer_list<vk::DeviceSize> const& offsets,
        uint32_t index = 0);

    void UpdateArgument_IndexBuffer(vk::Buffer buffer, vk::DeviceSize offset,
                                    vk::IndexType type, uint32_t index = 0);

    void UpdateArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                           vk::DeviceSize offset,
                                           uint32_t drawCount, uint32_t stride,
                                           uint32_t index = 0);

    void UpdateArgument_Draw(uint32_t vertexCount, uint32_t instanceCount,
                             uint32_t firstVertex, uint32_t firstInstance,
                             uint32_t index = 0);

    void UpdateArgument_DispatchIndirect(vk::Buffer buffer,
                                         vk::DeviceSize offset,
                                         uint32_t index = 0);

    void UpdateArgument_Dispatch(uint32_t x, uint32_t y, uint32_t z,
                                 uint32_t index = 0);

    void UpdateArgument_DrawMeshTasksIndirect(vk::Buffer buffer,
                                              vk::DeviceSize offset,
                                              uint32_t drawCount,
                                              uint32_t stride,
                                              uint32_t index = 0);

    void UpdateArgument_DrawMeshTask(uint32_t x, uint32_t y, uint32_t z,
                                     uint32_t index = 0);

    // index: -1 depth attachment
    //        0, 1, 2 ... color attachments
    void UpdateArgument_Attachments(
        Type_STLVector<int> const& indices,
        Type_STLVector<vk::RenderingAttachmentInfo> const& attachments);

    void UpdateArgument_ImageBarriers_BeforePass(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers = {},
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers = {},
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers = {});

    void UpdateArgument_ImageBarriers_AfterPass(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers = {},
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers = {},
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers = {});

public:
    void RecordCmd(vk::CommandBuffer cmd) const;

    void Clear();

private:
    void PushMetaDataMapping(DrawCallMetaDataType type);

    template <DrawCallMetaDataType Type>
    DrawCallMetaData<Type>& FindMetaDataRef(uint32_t index);

private:
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_BeforePass;
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>
        mRenderingInfo;
    Type_STLVector<Type_DrawCallMetaData_Unified> mMetaDatas {};
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_AfterPass;

    MetaDataResourceMapping mMapping;
};

template <DrawCallMetaDataType Type>
DrawCallMetaData<Type>& DrawCallManager::FindMetaDataRef(uint32_t index) {
    auto indices = mMapping.metaData.at(Type);
    assert(index < indices.size());

    return mMetaDatas[indices[index]].Get<Type>();
}

}  // namespace IntelliDesign_NS::Vulkan::Core