#include "Shader.hpp"

#include <stdexcept>

#include <shaderc/shaderc.hpp>
#include <spirv_glsl.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Vulkan/Manager/Context.hpp"

namespace {

using namespace IntelliDesign_NS::Core::MemoryPool;

class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
    shaderc_include_result* GetInclude(const char* requested_source,
                                       shaderc_include_type type,
                                       const char* requesting_source,
                                       size_t include_depth) override {
        const Type_STLString name = SHADER_PATH(requested_source);

        std::ifstream file(name.c_str(), std::ios::in);

        if (!file.is_open()) {
            throw ::std::runtime_error(
                (Type_STLString("Cannot open shader source file: ") + name)
                    .c_str());
        }

        ::std::ostringstream sstr;
        sstr << file.rdbuf();
        Type_STLString contents {sstr.str()};

        auto container = new std::array<Type_STLString, 2>;
        (*container)[0] = name;
        (*container)[1] = contents;

        auto data = new shaderc_include_result;

        data->user_data = container;

        data->source_name = (*container)[0].data();
        data->source_name_length = (*container)[0].size();

        data->content = (*container)[1].data();
        data->content_length = (*container)[1].size();

        return data;
    }

    void ReleaseInclude(shaderc_include_result* data) override {
        delete static_cast<std::array<Type_STLString, 2>*>(data->user_data);
        delete data;
    }
};

Type_STLVector<uint32_t> LoadSPIRVCode(const char* filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("Cannot open binary file: ") + filePath).c_str());
    }

    uint32_t fileSize = file.tellg();

    Type_STLVector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

Type_STLVector<uint32_t> CompileGLSLSource(
    const char* name, const char* filePath, vk::ShaderStageFlagBits stage,
    bool hasInclude,
    IntelliDesign_NS::Vulkan::Core::Type_ShaderMacros const& defines,
    const char* entry) {
    std::ifstream file(filePath, std::ios::in);

    if (!file.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("Cannot open shader source file: ") + filePath)
                .c_str());
    }

    ::std::ostringstream sstr;
    sstr << file.rdbuf();
    Type_STLString buffer {sstr.str()};

    shaderc::Compiler compiler {};
    shaderc::CompileOptions options {};

    for (auto& [macro, value] : defines) {
        options.AddMacroDefinition(::std::string(macro), ::std::string(value));
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
        case vk::ShaderStageFlagBits::eVertex:
            kind = shaderc_glsl_vertex_shader;
            break;
        case vk::ShaderStageFlagBits::eFragment:
            kind = shaderc_glsl_fragment_shader;
            break;
        case vk::ShaderStageFlagBits::eCompute:
            kind = shaderc_glsl_compute_shader;
            break;
        case vk::ShaderStageFlagBits::eTaskEXT:
            kind = shaderc_glsl_task_shader;
            break;
        case vk::ShaderStageFlagBits::eMeshEXT:
            kind = shaderc_glsl_mesh_shader;
            break;
        default:
            throw ::std::runtime_error(
                "ERROR::CompileGLSLSource: Invalid shader stage");
            break;
    }

    shaderc::SpvCompilationResult spirvModule {};
    if (hasInclude) {
        options.SetIncluder(::std::make_unique<ShaderIncluder>());
        auto preprocess =
            compiler.PreprocessGlsl(buffer.c_str(), kind, name, options);
        if (preprocess.GetCompilationStatus()
            != shaderc_compilation_status_success) {
            ::std::cerr << preprocess.GetErrorMessage();
            return {};
        }
        Type_STLString preprocessedSource {preprocess.begin()};

        spirvModule = compiler.CompileGlslToSpv(preprocessedSource.c_str(),
                                                kind, name, entry, options);
    } else {
        spirvModule = compiler.CompileGlslToSpv(buffer.c_str(), kind, name,
                                                entry, options);
    }

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
               vk::ShaderStageFlagBits stage, const char* entry, void* pNext)
    : pContext(context), mName(name), mEntry(entry), mStage(stage) {
    mSPIRVBinaryCode = LoadSPIRVCode(path);
    mShader = CreateShader(pNext);
    ReflectDescSetLayouts();
}

Shader::Shader(Context* context, const char* name, const char* sourcePath,
               vk::ShaderStageFlagBits stage, bool hasIncludes,
               Type_ShaderMacros const& defines, const char* entry, void* pNext)
    : pContext(context), mName(name), mEntry(entry), mStage(stage) {
    mSPIRVBinaryCode = CompileGLSLSource("", sourcePath, stage, hasIncludes,
                                         defines, mEntry.c_str());
    mShader = CreateShader(pNext);
    ReflectDescSetLayouts();
}

Shader::~Shader() {
    pContext->GetDeviceHandle().destroy(mShader);
}

vk::PipelineShaderStageCreateInfo Shader::GetStageInfo(void* pNext) const {
    vk::PipelineShaderStageCreateInfo info;
    info.setModule(mShader)
        .setStage(mStage)
        .setPName(mEntry.c_str())
        .setPNext(pNext);

    return info;
}

std::span<uint32_t> Shader::GetBinaryCode() {
    return mSPIRVBinaryCode;
}

std::span<DescriptorSetLayoutData> Shader::GetDescSetLayoutDatas() {
    return mDescSetLayoutDatas;
}

::std::optional<vk::PushConstantRange> const& Shader::GetPushContantData()
    const {
    return mPushContantData;
}

Type_STLString const& Shader::GetName() const {
    return mName;
}

Type_STLVector<vk::DescriptorSetLayout>& Shader::GetAllDescSetLayoutHandles() {
    return mDescSetLayouts;
}

std::mutex& Shader::GetMutex() {
    return mMutex;
}

vk::ShaderModule Shader::CreateShader(void* pNext) const {
    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(mSPIRVBinaryCode).setPNext(pNext);

    return pContext->GetDeviceHandle().createShaderModule(createInfo);
}

void Shader::ReflectDescSetLayouts() {
    spirv_cross::CompilerGLSL compiler {mSPIRVBinaryCode.data(),
                                        mSPIRVBinaryCode.size()};

    auto resources = compiler.get_shader_resources();

    auto parseFunc = [&](spirv_cross::Resource const& resource,
                         vk::DescriptorType type) {
        Type_STLString name {resource.name};
        uint32_t setIdx =
            compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        uint32_t binding =
            compiler.get_decoration(resource.id, spv::DecorationBinding);
        uint32_t descCount = 1;  // TODO: Parse Desc Count
        mDescSetLayoutDatas.emplace_back(name.c_str(), setIdx, binding, type,
                                         mStage, descCount);
    };

    // Uniform buffers
    for (auto& resource : resources.uniform_buffers) {
        parseFunc(resource, vk::DescriptorType::eUniformBuffer);
    }

    // Storage buffers
    for (auto& resource : resources.storage_buffers) {
        parseFunc(resource, vk::DescriptorType::eStorageBuffer);
    }

    // Storage images
    for (auto& resource : resources.storage_images) {
        parseFunc(resource, vk::DescriptorType::eStorageImage);
    }

    // combined image sampler
    for (auto& resource : resources.sampled_images) {
        parseFunc(resource, vk::DescriptorType::eCombinedImageSampler);
    }

    // push constant
    for (auto& resource : resources.push_constant_buffers) {
        const spirv_cross::SPIRType& type =
            compiler.get_type(resource.base_type_id);
        vk::PushConstantRange pushConstant;
        pushConstant.size = compiler.get_declared_struct_size(type);
        pushConstant.stageFlags = mStage;
        pushConstant.offset = 0;
        mPushContantData.emplace(pushConstant);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core