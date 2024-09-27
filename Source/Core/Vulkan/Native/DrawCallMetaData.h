#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/System/CommonBasedVariant.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

enum class DrawCallMetaDataType {
    ClearColorImage,
    ClearDepthStencilImage,
    MemoryBarrier,
    RenderingInfo,
    Viewport,
    Scissor,
    Pipeline,
    PushContant,
    DescriptorBuffer,
    DescriptorSet,
    IndexBuffer,
    DrawIndexedIndirect,
    Draw,
    DispatchIndirect,
    Dispatch,
    DrawMeshTasksIndirect,
    DrawMeshTask,

    NumTypes
};

struct RenderingAttachmentInfo {
    Type_STLString imageName;
    Type_STLString viewName;
    vk::RenderingAttachmentInfo info;
};

class RenderResourceManager;

struct IDrawCallMetaData {
    virtual void UpdateRenderResource(RenderResourceManager* manager,
                                      Type_STLString name) {}

    virtual void RecordCmds(vk::CommandBuffer cmd) const = 0;
};

template <DrawCallMetaDataType Type>
struct DrawCallMetaData;

template <>
struct DrawCallMetaData<DrawCallMetaDataType::ClearColorImage>
    : IDrawCallMetaData {
    vk::Image image;
    vk::ImageLayout layout;
    vk::ClearColorValue clearValue;
    Type_STLVector<vk::ImageSubresourceRange> ranges;

    void UpdateRenderResource(RenderResourceManager* manager,
                              Type_STLString name) override;
    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::ClearDepthStencilImage>
    : IDrawCallMetaData {
    vk::Image image;
    vk::ImageLayout layout;
    vk::ClearDepthStencilValue clearValue;
    Type_STLVector<vk::ImageSubresourceRange> ranges;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::MemoryBarrier>
    : IDrawCallMetaData {
    ::std::optional<Type_STLVector<vk::ImageMemoryBarrier2>> imgBarriers {};
    ::std::optional<Type_STLVector<vk::MemoryBarrier2>> memBarriers {};
    ::std::optional<Type_STLVector<vk::BufferMemoryBarrier2>> bufBarriers {};

    Type_STLUnorderedMap_String<
        ::std::variant<vk::ImageMemoryBarrier2*, vk::MemoryBarrier2*,
                       vk::BufferMemoryBarrier2*>>
        mapping {};

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::RenderingInfo>
    : IDrawCallMetaData {
    Type_STLVector<vk::RenderingAttachmentInfo> colorAttachments;
    ::std::optional<vk::RenderingAttachmentInfo> depthStencilAttachment;

    vk::RenderingInfo info;

    // <image name>@<view name> - index
    // index : 0, 1, 2 ... color attachments
    //        -1 depth attachment
    Type_STLUnorderedMap_String<int> mapping {};

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Viewport> : IDrawCallMetaData {
    uint32_t firstViewport {0};
    Type_STLVector<vk::Viewport> viewports {};

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Scissor> : IDrawCallMetaData {
    uint32_t firstScissor {0};
    Type_STLVector<vk::Rect2D> scissors {};

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Pipeline> : IDrawCallMetaData {
    vk::PipelineBindPoint bindPoint;
    vk::Pipeline pipeline;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::PushContant> : IDrawCallMetaData {
    vk::PipelineLayout layout;
    vk::ShaderStageFlags stage;
    uint32_t offset;
    uint32_t size;
    const void* pValues;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DescriptorBuffer>
    : IDrawCallMetaData {
    Type_STLVector<vk::DeviceAddress> addresses;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DescriptorSet>
    : IDrawCallMetaData {
    vk::PipelineBindPoint bindPoint;
    vk::PipelineLayout layout;
    uint32_t firstSet;
    Type_STLVector<uint32_t> bufferIndices;
    Type_STLVector<vk::DeviceSize> offsets;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::IndexBuffer> : IDrawCallMetaData {
    vk::Buffer buffer;
    vk::DeviceSize offset;
    vk::IndexType type;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DrawIndexedIndirect>
    : IDrawCallMetaData {
    vk::Buffer buffer;
    vk::DeviceSize offset;
    uint32_t drawCount;
    uint32_t stride;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Draw> : IDrawCallMetaData {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DispatchIndirect>
    : IDrawCallMetaData {
    vk::Buffer buffer;
    vk::DeviceSize offset;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Dispatch> : IDrawCallMetaData {
    uint32_t x, y, z;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DrawMeshTasksIndirect>
    : IDrawCallMetaData {
    vk::Buffer buffer;
    vk::DeviceSize offset;
    uint32_t drawCount;
    uint32_t stride;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DrawMeshTask>
    : IDrawCallMetaData {
    uint32_t x, y, z;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

INTELLI_DS_DEFINE_ENUM_INDEXED_CommonBasedVariant(
    Type_DrawCallMetaData_Unified, IDrawCallMetaData, DrawCallMetaData,
    DrawCallMetaDataType, DrawCallMetaDataType::NumTypes);

}  // namespace IntelliDesign_NS::Vulkan::Core