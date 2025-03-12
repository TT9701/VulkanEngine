#pragma once

#include <optional>

#include <vulkan/vulkan.hpp>

#include "Core/System/CommonBasedVariant.hpp"
#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

struct DGCColorBlendInfo {
    uint32_t firstAttachment {0};
    Type_STLVector<vk::Bool32> enableColorBlend {vk::False};
    Type_STLVector<vk::ColorBlendEquationEXT> equations {1};
};

struct DGCPipelineInfo {
    vk::PolygonMode polygonMode {vk::PolygonMode::eFill};
    vk::CullModeFlags cullMode {vk::CullModeFlagBits::eNone};
    vk::SampleCountFlagBits rasterSampleCount {vk::SampleCountFlagBits::e1};
    vk::Bool32 enableDepthTest {vk::True};
    vk::Bool32 enableDepthWrite {vk::True};
    vk::CompareOp depthCompareOp {vk::CompareOp::eGreaterOrEqual};
    DGCColorBlendInfo colorBlendInfo {};
    vk::Viewport viewport {};
    vk::Rect2D scissor {};
};

enum class DrawCallMetaDataType {
    ClearColorImage,
    ClearDepthStencilImage,

    MemoryBarrier,

    RenderingInfo,
    Viewport,
    Scissor,

    Pipeline,
    DescriptorBuffer,
    DescriptorSet,

    DGCSequence,
    DGCPipelineInfo,  ///<- only used when execution set is shaderEXT, and graphic draw

    Copy,

    NumTypes
};

struct RenderingAttachmentInfo {
    Type_STLString imageName;
    Type_STLString viewName;
    vk::RenderingAttachmentInfo info {};
};

class RenderResource;
class RenderResourceManager;

struct IDrawCallMetaData {
    virtual void UpdateRenderResource(RenderResourceManager& manager,
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

    void UpdateRenderResource(RenderResourceManager& manager,
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
struct DrawCallMetaData<DrawCallMetaDataType::DGCSequence> : IDrawCallMetaData {
    RenderResource const* sequenceBuffer;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::DGCPipelineInfo>
    : IDrawCallMetaData {
    DGCPipelineInfo pipelineInfo;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

template <>
struct DrawCallMetaData<DrawCallMetaDataType::Copy> : IDrawCallMetaData {
    enum class Type {
        BufferToBuffer,
        BufferToImage,
        ImageToImage,
        ImageToBuffer
    };

    using Type_CopyRegion =
        ::std::variant<vk::BufferCopy2, vk::BufferImageCopy2, vk::ImageCopy2>;
    using Type_CopyInfo =
        ::std::variant<vk::CopyBufferInfo2, vk::CopyBufferToImageInfo2,
                       vk::CopyImageInfo2, vk::CopyImageToBufferInfo2>;

    Type_CopyInfo info;

    void RecordCmds(vk::CommandBuffer cmd) const override;
};

INTELLI_DS_DEFINE_ENUM_INDEXED_CommonBasedVariant(
    Type_DrawCallMetaData_Unified, IDrawCallMetaData, DrawCallMetaData,
    DrawCallMetaDataType, DrawCallMetaDataType::NumTypes);

}  // namespace IntelliDesign_NS::Vulkan::Core