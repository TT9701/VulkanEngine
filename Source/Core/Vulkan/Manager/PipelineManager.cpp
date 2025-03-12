#include "PipelineManager.h"

#include <spirv_glsl.hpp>

#include "Core/Vulkan/Native/Shader.h"
#include "VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

PipelineManager::PipelineManager(VulkanContext& contex) : mContext(contex) {}

PipelineLayout* PipelineManager::CreateLayout(
    const char* name, ShaderProgram* program,
    vk::PipelineLayoutCreateFlags flags, void* pNext) {
    const auto ptr =
        MakeShared<PipelineLayout>(mContext, *program, flags, pNext);

    mContext.SetName(ptr->GetHandle(), name);
    mPipelineLayouts.emplace(name, ptr);

    return ptr.get();
}

vk::PipelineLayout PipelineManager::GetLayoutHandle(const char* name) const {
    return GetLayout(name)->GetHandle();
}

PipelineLayout* PipelineManager::GetLayout(const char* name) const {
    return mPipelineLayouts.at(ParsePipelineLayoutName(name)).get();
}

vk::Pipeline PipelineManager::GetPipelineHandle(const char* name) const {
    return GetPipeline(name).GetHandle();
}

Pipeline& PipelineManager::GetPipeline(const char* name) const {
    return *mPipelines.at(ParsePipelineName(name));
}

PipelineManager::Type_Pipelines const& PipelineManager::GetPipelines() const {
    return mPipelines;
}

PipelineManager::Type_CPBuilder PipelineManager::GetBuilder_Compute() {
    return Type_CPBuilder {*this};
}

PipelineManager::Type_GPBuilder PipelineManager::GetBuilder_Graphics() {
    return Type_GPBuilder {*this};
}

void PipelineManager::BindPipeline(vk::CommandBuffer cmd, const char* name) {
    auto& pipeline = GetPipeline(name);
    if (pipeline.mType == PipelineType::Compute) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.mHandle);
    } else if (pipeline.mType == PipelineType::Graphics) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.mHandle);
    }
}

Type_STLString PipelineManager::ParsePipelineName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName};
}

Type_STLString PipelineManager::ParsePipelineLayoutName(
    const char* pipelineName) const {
    return Type_STLString {pipelineName};
}

}  // namespace IntelliDesign_NS::Vulkan::Core