#pragma once

#include "Core/Vulkan/Native/DrawCallMetaData.h"

namespace IntelliDesign_NS::Vulkan::Core {

class RenderResourceManager;

class DrawCallManager {
public:
    DrawCallManager(RenderResourceManager* manager);

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

    void AddArgument_Barriers_BeforePass(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers = {},
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers = {},
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers = {});

    void AddArgument_Barriers_AfterPass(
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

    void AddArgument_PushConstant(vk::PipelineLayout layout,
                                  vk::ShaderStageFlags stage, uint32_t offset,
                                  uint32_t size, const void* pValues);

    void AddArgument_DescriptorBuffer(
        Type_STLVector<vk::DeviceAddress> const& addresses);

    void AddArgument_DescriptorSet(
        vk::PipelineBindPoint bindPoint, vk::PipelineLayout layout,
        uint32_t firstSet, Type_STLVector<uint32_t> const& bufferIndices,
        Type_STLVector<vk::DeviceSize> const& offsets);

    void AddArgument_IndexBuffer(vk::Buffer buffer, vk::DeviceSize offset,
                                 vk::IndexType type);

    void AddArgument_DrawIndexedIndiret(vk::Buffer buffer,
                                        vk::DeviceSize offset,
                                        uint32_t drawCount, uint32_t stride);

    void AddArgument_DrawIndiret(vk::Buffer buffer,
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

    void UpdateArgument_Viewport(uint32_t firstViewport,
                                 Type_STLVector<vk::Viewport> const& viewports,
                                 uint32_t index = 0);

    void UpdateArgument_Scissor(uint32_t firstScissor,
                                Type_STLVector<vk::Rect2D> const& scissors,
                                uint32_t index = 0);

    void UpdateArgument_Pipeline(vk::PipelineBindPoint bindPoint,
                                 vk::Pipeline pipeline, uint32_t index = 0);

    void UpdateArgument_PushConstant(vk::PipelineLayout layout,
                                     vk::ShaderStageFlags stage,
                                     uint32_t offset, uint32_t size,
                                     const void* pValues, uint32_t index = 0);

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

    // name = imageName@viewName
    void UpdateArgument_Attachments(
        Type_STLVector<Type_STLString> const& names);

    void UpdateArgument_Attachments(
        Type_STLVector<int> const& indices,
        Type_STLVector<vk::RenderingAttachmentInfo> const& attachments);

    void UpdateArgument_Barriers_BeforePass(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers);

    // only update resources
    void UpdateArgument_Barriers_BeforePass(
        Type_STLVector<Type_STLString> const& names);

    void UpdateArgument_Barriers_AfterPass(
        Type_STLVector<Type_STLString> const& names,
        Type_STLVector<vk::ImageMemoryBarrier2> const& imgBarriers,
        Type_STLVector<vk::MemoryBarrier2> const& memBarriers,
        Type_STLVector<vk::BufferMemoryBarrier2> const& bufBarriers);

    // only update resources
    void UpdateArgument_Barriers_AfterPass(
        Type_STLVector<Type_STLString> const& names);

    void UpdateArgument_OnResize(vk::Extent2D extent);

public:
    void RecordCmd(vk::CommandBuffer cmd) const;

    void Clear();

private:
    template <DrawCallMetaDataType Type>
    DrawCallMetaData<Type>* FindMetaDataPtr(uint32_t index = 0);

    void PushResourceMetaDataMapping(const char* name, uint32_t index);

private:
    RenderResourceManager* pRenderResManager;

    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_BeforePass;
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>>
        mRenderingInfo;
    Type_STLVector<Type_DrawCallMetaData_Unified> mMetaDatas {};
    ::std::optional<DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>>
        mBarriers_AfterPass;

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