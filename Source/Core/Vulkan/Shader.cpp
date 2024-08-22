#include "Shader.hpp"

#include <stdexcept>

#include "Context.hpp"

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

namespace IntelliDesign_NS::Vulkan::Core {

Shader::Shader(Context* context, std::string name,
               std::vector<uint32_t> const& binaryCode, ShaderStage stage,
               std::string entry, void* pNext)
    : pContext(context),
      mName(std::move(name)),
      mEntry(std::move(entry)),
      mStage(stage),
      mShaderModule(CreateShaderModule(binaryCode, pNext)) {}

Shader::Shader(Context* context, ::std::string const& name,
               std::string const& path, ShaderStage stage,
               std::string const& entry, void* pNext)
    : Shader(context, name, LoadSPIRVCode(path), stage, entry, pNext) {}

Shader::~Shader() {
    pContext->GetDeviceHandle().destroy(mShaderModule);
}

vk::PipelineShaderStageCreateInfo Shader::GetStageInfo(void* pNext) const {
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
        case ShaderStage::Task:
            stage = vk::ShaderStageFlagBits::eTaskEXT;
            break;
        case ShaderStage::Mesh:
            stage = vk::ShaderStageFlagBits::eMeshEXT;
            break;
        default: throw std::runtime_error("Unfinished shader stage!");
    }
    info.setModule(mShaderModule)
        .setStage(stage)
        .setPName(mEntry.c_str())
        .setPNext(pNext);

    return info;
}

vk::ShaderModule Shader::CreateShaderModule(
    ::std::vector<uint32_t> const& binaryCode, void* pNext) const {
    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(binaryCode).setPNext(pNext);

    return pContext->GetDeviceHandle().createShaderModule(createInfo);
}

}  // namespace IntelliDesign_NS::Vulkan::Core