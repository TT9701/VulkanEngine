#include "Shader.hpp"

#include <stdexcept>

#include "Context.hpp"

namespace {

std::vector<uint32_t> LoadSPIRVCode(const char* filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw ::std::runtime_error(::std::string("Cannot open binary file: ")
                                   + filePath);
    }

    uint32_t fileSize = file.tellg();

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

std::vector<uint32_t> CompileGLSLSource(
    const char* name, const char* filePath,
    IntelliDesign_NS::Vulkan::Core::ShaderStage stage,
    ::std::unordered_map<::std::string, ::std::string> const& defines,
    const char* entry) {
    std::ifstream file(filePath, std::ios::ate);

    if (!file.is_open()) {
        throw ::std::runtime_error(
            ::std::string("Cannot open shader source file: ") + filePath);
    }

    ::std::ostringstream sstr;
    sstr << file.rdbuf();
    auto buffer = sstr.str();

    shaderc::Compiler compiler {};
    shaderc::CompileOptions options {};

    for (auto& [macro, value] : defines) {
        options.AddMacroDefinition(macro, value);
    }
#ifndef NDEBUG
    options.SetGenerateDebugInfo();
#else
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif
    options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                 shaderc_env_version_vulkan_1_3);
    options.SetTargetSpirv(shaderc_spirv_version_1_4);

    shaderc_shader_kind kind {};

    switch (stage) {
        case IntelliDesign_NS::Vulkan::Core::ShaderStage::Vertex:
            kind = shaderc_glsl_vertex_shader;
            break;
        case IntelliDesign_NS::Vulkan::Core::ShaderStage::Fragment:
            kind = shaderc_glsl_fragment_shader;
            break;
        case IntelliDesign_NS::Vulkan::Core::ShaderStage::Compute:
            kind = shaderc_glsl_compute_shader;
            break;
        case IntelliDesign_NS::Vulkan::Core::ShaderStage::Task:
            kind = shaderc_glsl_task_shader;
            break;
        case IntelliDesign_NS::Vulkan::Core::ShaderStage::Mesh:
            kind = shaderc_glsl_mesh_shader;
            break;
        default:
            throw ::std::runtime_error(
                "ERROR::CompileGLSLSource: Invalid shader stage");
            break;
    }

    auto spirvModule =
        compiler.CompileGlslToSpv(buffer, kind, name, entry, options);

    if (spirvModule.GetCompilationStatus()
        != shaderc_compilation_status_success) {
        ::std::cerr << spirvModule.GetErrorMessage();
        return {};
    }
    return {spirvModule.cbegin(), spirvModule.cend()};
}

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

Shader::Shader(Context* context, const char* name, const char* path,
               ShaderStage stage, const char* entry, void* pNext)
    : pContext(context), mName(name), mEntry(entry), mStage(stage) {
    auto binarycode = LoadSPIRVCode(path);
    mShaderModule = CreateShaderModule(binarycode, pNext);
    pContext->SetName(mShaderModule, name);
}

Shader::Shader(Context* context, const char* name, const char* sourcePath,
               ShaderStage stage,
               std::unordered_map<std::string, std::string> const& defines,
               const char* entry, void* pNext)
    : pContext(context), mName(name), mEntry(entry), mStage(stage) {
    auto binaryCode =
        CompileGLSLSource(name, sourcePath, stage, defines, mEntry.c_str());
    mShaderModule = CreateShaderModule(binaryCode, pNext);
    pContext->SetName(mShaderModule, name);
}

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

vk::ShaderModule Shader::CreateShaderModule(::std::span<uint32_t> binaryCode,
                                            void* pNext) const {
    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(binaryCode).setPNext(pNext);

    return pContext->GetDeviceHandle().createShaderModule(createInfo);
}

}  // namespace IntelliDesign_NS::Vulkan::Core