#include "VulkanShader.hpp"

#include <stdexcept>

#include "VulkanContext.hpp"

namespace {

std::vector<uint32_t> LoadSPIRVCode(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw ::std::runtime_error("Cannot open binary file: " + filePath);
    }

    uint32_t fileSize = file.tellg();

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

}  // namespace

VulkanShader::VulkanShader(VulkanContext* context, std::string name,
                           std::vector<uint32_t> const& binaryCode,
                           ShaderStage stage, std::string entry, void* pNext)
    : pContext(context),
      mName(std::move(name)),
      mEntry(std::move(entry)),
      mStage(stage),
      mShaderModule(CreateShaderModule(binaryCode, pNext)) {}

VulkanShader::VulkanShader(VulkanContext* context, ::std::string const& name,
                           std::string const& path, ShaderStage stage,
                           std::string const& entry, void* pNext)
    : VulkanShader(context, name, LoadSPIRVCode(path), stage, entry, pNext) {}

VulkanShader::~VulkanShader() {
    pContext->GetDeviceHandle().destroy(mShaderModule);
}

vk::PipelineShaderStageCreateInfo VulkanShader::GetStageInfo(
    void* pNext) const {
    vk::PipelineShaderStageCreateInfo info;

    vk::ShaderStageFlagBits stage;
    switch (mStage) {
        case ShaderStage::Compute:
            stage = vk::ShaderStageFlagBits::eCompute;
            break;
        case ShaderStage::Vertex:
            stage = vk::ShaderStageFlagBits::eVertex;
            break;
        case ShaderStage::Fragment:
            stage = vk::ShaderStageFlagBits::eFragment;
            break;
        default: throw std::runtime_error("Unfinished shader stage!");
    }
    info.setModule(mShaderModule)
        .setStage(stage)
        .setPName(mEntry.c_str())
        .setPNext(pNext);

    return info;
}

vk::ShaderModule VulkanShader::CreateShaderModule(
    ::std::vector<uint32_t> const& binaryCode, void* pNext) const {
    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(binaryCode).setPNext(pNext);

    return pContext->GetDeviceHandle().createShaderModule(createInfo);
}