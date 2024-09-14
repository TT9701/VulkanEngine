#pragma once

#include "Core/Vulkan/Native/DrawCallMetaData.h"

namespace IntelliDesign_NS::Vulkan::Core {

class DrawCallManager {
public:
    void AddArgument_ClearColorImage(
        vk::Image image, vk::ImageLayout layout,
        vk::ClearColorValue const& clearValue,
        ::std::initializer_list<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_ClearDepthStencilImage(
        vk::Image image, vk::ImageLayout layout,
        vk::ClearDepthStencilValue const& clearValue,
        ::std::initializer_list<vk::ImageSubresourceRange> const& ranges);

    void AddArgument_MemoryBarriers_BeforePass(
        ::std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers =
            {},
        ::std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers =
            {},
        ::std::initializer_list<vk::MemoryBarrier2> const& memBarriers = {});

    void AddArgument_MemoryBarriers_AfterPass(
        ::std::initializer_list<vk::ImageMemoryBarrier2> const& imgBarriers =
            {},
        ::std::initializer_list<vk::BufferMemoryBarrier2> const& bufBarriers =
            {},
        ::std::initializer_list<vk::MemoryBarrier2> const& memBarriers = {});

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

    void UpdateArgument_ColorAttachments(
        ::std::initializer_list<vk::RenderingAttachmentInfo> const&
            colorAttachments);

public:
    void RecordCmd(vk::CommandBuffer cmd) const;

private:
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_BeforePass;
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>
        mRenderingInfo;
    Type_STLVector<Type_DrawCallMetaData_Unified> mMetaDatas {};
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_AfterPass;
};
}  // namespace IntelliDesign_NS::Vulkan::Core