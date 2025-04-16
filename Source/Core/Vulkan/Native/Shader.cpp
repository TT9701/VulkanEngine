#include "Shader.h"

#include <regex>
#include <sstream>
#include <stdexcept>

#include <shaderc/shaderc.hpp>
#include <spirv_glsl.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Vulkan/Manager/VulkanContext.h"

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
                (Type_STLString("Cannot open shader source file: ")
                 + name.c_str())
                    .c_str());
        }

        ::std::ostringstream sstr;
        sstr << file.rdbuf();
        Type_STLString contents {sstr.str().c_str()};

        auto container = new std::array<Type_STLString, 2>;
        (*container)[0] = name;
        (*container)[1] = contents;

        auto data = new shaderc_include_result;

        data->user_data = container;

        data->source_name = (*container)[0].c_str();
        data->source_name_length = (*container)[0].size();

        data->content = (*container)[1].c_str();
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

Type_ANSIString LoadShaderSource(const char* filePath) {
    std::ifstream file(filePath, std::ios::in);

    if (!file.is_open()) {
        throw ::std::runtime_error(
            (Type_STLString("Cannot open shader source file: ") + filePath)
                .c_str());
    }

    ::std::ostringstream sstr;
    sstr << file.rdbuf();
    return Type_ANSIString {sstr.str().c_str()};
}

Type_STLVector<uint32_t> CompileGLSLSource(
    const char* name, Type_ANSIString const& source,
    vk::ShaderStageFlagBits stage, bool hasInclude,
    IntelliDesign_NS::Vulkan::Core::Type_ShaderMacros const& defines,
    const char* entry) {
    shaderc::Compiler compiler {};
    shaderc::CompileOptions options {};

    for (auto& [macro, value] : defines) {
        options.AddMacroDefinition(::std::string(macro.c_str()),
                                   ::std::string(value.c_str()));
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
            compiler.PreprocessGlsl(source.c_str(), kind, name, options);
        if (preprocess.GetCompilationStatus()
            != shaderc_compilation_status_success) {
            ::std::cerr << preprocess.GetErrorMessage();
            return {};
        }
        Type_STLString preprocessedSource {preprocess.begin()};

        spirvModule = compiler.CompileGlslToSpv(preprocessedSource.c_str(),
                                                kind, name, entry, options);
    } else {
        spirvModule = compiler.CompileGlslToSpv(source.c_str(), kind, name,
                                                entry, options);
    }

    if (spirvModule.GetCompilationStatus()
        != shaderc_compilation_status_success) {
        ::std::cerr << spirvModule.GetErrorMessage();
        return {};
    }
    return {spirvModule.cbegin(), spirvModule.cend()};
}

Type_STLVector<uint32_t> CompileGLSLSource(
    const char* name, const char* filePath, vk::ShaderStageFlagBits stage,
    bool hasInclude,
    IntelliDesign_NS::Vulkan::Core::Type_ShaderMacros const& defines,
    const char* entry) {
    auto buffer = LoadShaderSource(filePath);
    return CompileGLSLSource(name, buffer, stage, hasInclude, defines, entry);
}

using namespace IntelliDesign_NS::Vulkan::Core;

Type_STLVector<ShaderBase::DescriptorSetLayoutData> MergeDescLayoutData(
    Type_STLVector<ShaderBase*> const& shaders) {
    Type_STLVector<ShaderBase::DescriptorSetLayoutData> datas {};

    for (auto const& shader : shaders) {
        if (shader)
            datas.insert(datas.end(), shader->GetDescSetLayoutDatas().begin(),
                         shader->GetDescSetLayoutDatas().end());
    }

    if (datas.empty())
        return datas;

    std::ranges::sort(datas, [](ShaderBase::DescriptorSetLayoutData const& l,
                                ShaderBase::DescriptorSetLayoutData const& r) {
        if (l.setIdx != r.setIdx) {
            return l.setIdx < r.setIdx;
        }
        return l.bindingIdx < r.bindingIdx;
    });

    auto makeUniqueSet =
        [&](const Type_STLVector<ShaderBase::DescriptorSetLayoutData>::iterator&
                prev,
            const Type_STLVector<ShaderBase::DescriptorSetLayoutData>::iterator&
                last) {
            Type_STLVector<ShaderBase::DescriptorSetLayoutData> uniqueSet {};
            for (auto it = prev; it != last; ++it) {
                uniqueSet.push_back(*it);
            }
            return uniqueSet;
        };

    Type_STLVector<Type_STLVector<ShaderBase::DescriptorSetLayoutData>>
        uniqueSets {};
    {
        auto prev = datas.begin();
        auto last = ++datas.begin();

        while (prev != datas.end()) {
            if (last == datas.end()) {
                uniqueSets.push_back(makeUniqueSet(prev, last));
                break;
            }
            if (last->setIdx == prev->setIdx) {
                ++last;
                continue;
            }
            uniqueSets.push_back(makeUniqueSet(prev, last));
            prev = last;
        }
    }

    auto mergeBinding =
        [&](Type_STLVector<ShaderBase::DescriptorSetLayoutData> const&
                bindings) {
            ShaderBase::DescriptorSetLayoutData data {bindings[0]};
            Type_STLString prefix {};
            for (uint32_t i = 1; i < bindings.size(); ++i) {
                data.stage |= bindings[i].stage;
            }
            return data;
        };

    Type_STLVector<Type_STLVector<ShaderBase::DescriptorSetLayoutData>>
        uniqueBindingSets {};
    for (auto& set : uniqueSets) {
        auto prev = set.begin();
        auto last = ++set.begin();
        Type_STLVector<ShaderBase::DescriptorSetLayoutData> uniqueBindingSet {};
        while (prev != set.end()) {
            if (last == set.end()) {
                uniqueBindingSet.push_back(
                    mergeBinding(makeUniqueSet(prev, last)));

                break;
            }
            if (last->bindingIdx == prev->bindingIdx) {
                ++last;
                continue;
            }
            uniqueBindingSet.push_back(mergeBinding(makeUniqueSet(prev, last)));

            prev = last;
        }
        uniqueBindingSets.push_back(uniqueBindingSet);
    }

    datas.clear();
    for (auto& set : uniqueBindingSets) {
        for (auto& data : set) {
            datas.emplace_back(data);
        }
    }
    return datas;
}

Type_STLVector<SharedPtr<DescriptorSetLayout>> CreateDescLayout(
    VulkanContext& context,
    Type_STLVector<ShaderBase::DescriptorSetLayoutData> const& datas,
    void* pNext) {
    Type_STLVector<SharedPtr<DescriptorSetLayout>> descLayouts {};

    if (datas.empty())
        return descLayouts;

    auto descBufProps =
        context.GetPhysicalDevice()
            .GetProperties<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();

    auto CreateDescLayout =
        [&](const char* name,
            Type_STLVector<Type_STLString> const& bindingNames,
            Type_STLVector<vk::DescriptorSetLayoutBinding> const& bindings) {
            auto ptr = MakeShared<DescriptorSetLayout>(
                context, bindingNames, bindings, descBufProps, pNext);

            context.SetName(ptr->GetHandle(), name);

            descLayouts.push_back(ptr);
        };

    auto it = datas.begin();
    for (uint32_t setIdx = 0; setIdx <= datas.rbegin()->setIdx; ++setIdx) {
        Type_STLString setName {};
        Type_STLVector<Type_STLString> bindingNames;
        Type_STLVector<vk::DescriptorSetLayoutBinding> bindings;
        auto stage = vk::to_string(it->stage);
        std::erase(stage, ' ');
        setName = setName + "@" + stage.c_str();
        for (; it != datas.end() && it->setIdx == setIdx; ++it) {
            Type_STLString bindingName {it->name};
            bindingName = bindingName + "@" + stage.c_str();
            bindingNames.emplace_back(bindingName);
            bindings.emplace_back(it->bindingIdx, it->type, it->descCount,
                                  it->stage, nullptr);
        }
        setName = setName + "@Set" + ::std::to_string(setIdx).c_str();
        CreateDescLayout(setName.c_str(), bindingNames, bindings);
    }

    return descLayouts;
}

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

ShaderBase::ShaderBase(VulkanContext& context, const char* name,
                       const char* spirvPath, vk::ShaderStageFlagBits stage,
                       const char* entry)
    : mContext(context), mName(name), mEntry(entry), mStage(stage) {
    mSPIRVBinaryCode = LoadSPIRVCode(spirvPath);
    SPIRVReflect_DescSetLayouts();
    SPIRVReflect_PushContants();
}

ShaderBase::ShaderBase(VulkanContext& context, const char* name,
                       const char* sourcePath, vk::ShaderStageFlagBits stage,
                       bool hasIncludes, Type_ShaderMacros const& defines,
                       const char* entry)
    : mContext(context), mName(name), mEntry(entry), mStage(stage) {
    auto source = LoadShaderSource(sourcePath);
    GLSLReflect(source);
    mSPIRVBinaryCode = CompileGLSLSource("", source, stage, hasIncludes,
                                         defines, mEntry.c_str());
    SPIRVReflect_DescSetLayouts();
    SPIRVReflect_PushContants();
}

Type_STLVector<ShaderBase::DescriptorSetLayoutData>
ShaderBase::SPIRVReflect_DescSetLayouts() {
    spirv_cross::CompilerGLSL compiler {mSPIRVBinaryCode.data(),
                                        mSPIRVBinaryCode.size()};

    auto resources = compiler.get_shader_resources();

    Type_STLVector<DescriptorSetLayoutData> datas {};

    auto parseFunc = [&](spirv_cross::Resource const& resource,
                         vk::DescriptorType type) {
        uint32_t setIdx =
            compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        uint32_t binding =
            compiler.get_decoration(resource.id, spv::DecorationBinding);
        uint32_t descCount = 1;

        Type_STLString name {};
        for (auto const& [setBinding, v] : mGLSL_SetBindingNameMap) {
            if (setBinding.set == setIdx && setBinding.binding == binding) {
                name = v;
            }
        }

        const spirv_cross::SPIRType& spirtype =
            compiler.get_type(resource.type_id);

        // array
        if (!spirtype.array.empty()) {
            descCount = spirtype.array[0];

            // uniform sampler2D tex[], size == 0;
            if (descCount == 0) {
                descCount = 128;
                // descCount = ::std::min(
                //     MAX_BINDLESS_DESCRIPTOR_COUNT,
                //     mContext->GetDescIndexingProps()
                //         .maxDescriptorSetUpdateAfterBindSampledImages);
            }
        }

        datas.emplace_back(name, setIdx, binding, type, mStage, descCount);
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

    mDescSetLayoutDatas = datas;

    return datas;
}

std::optional<vk::PushConstantRange> ShaderBase::SPIRVReflect_PushContants() {
    spirv_cross::CompilerGLSL compiler {mSPIRVBinaryCode.data(),
                                        mSPIRVBinaryCode.size()};

    auto resources = compiler.get_shader_resources();

    ::std::optional<vk::PushConstantRange> data;

    // only one push contant
    for (auto& resource : resources.push_constant_buffers) {
        const spirv_cross::SPIRType& type =
            compiler.get_type(resource.base_type_id);
        vk::PushConstantRange pushConstant;
        pushConstant.size = compiler.get_declared_struct_size(type);
        pushConstant.stageFlags = mStage;
        auto ranges = compiler.get_active_buffer_ranges(resource.id);
        for (auto const& range : ranges) {
            auto index = range.index;
            auto offset = range.offset;
            auto size = range.range;
            if (index == 0) {
                pushConstant.offset = offset;
            }
        }
        pushConstant.size -= pushConstant.offset;
        data = pushConstant;
    }

    mPushContantData = data;

    return data;
}

std::span<uint32_t> ShaderBase::GetBinaryCode() {
    return mSPIRVBinaryCode;
}

std::span<ShaderBase::DescriptorSetLayoutData>
ShaderBase::GetDescSetLayoutDatas() {
    return mDescSetLayoutDatas;
}

std::optional<vk::PushConstantRange> const& ShaderBase::GetPushContantData()
    const {
    return mPushContantData;
}

Type_STLString const& ShaderBase::GetName() const {
    return mName;
}

std::mutex& ShaderBase::GetMutex() {
    return mMutex;
}

void ShaderBase::GLSLReflect(Type_ANSIString const& source) {
    auto fullsource = source;
    {
        Type_STLVector<Type_ANSIString> includeFileNames {};
        std::regex includePattern("#include\\s+\"([^\"]*)\"");

        auto includesBegin =
            std::sregex_iterator(source.begin(), source.end(), includePattern);
        auto includesEnd = std::sregex_iterator();

        for (std::sregex_iterator i = includesBegin; i != includesEnd; ++i) {
            std::smatch match = *i;
            includeFileNames.push_back(match.str(1).c_str());
        }

        for (auto const& name : includeFileNames) {
            fullsource += LoadShaderSource(SHADER_PATH_CSTR(name.c_str()));
        }
    }

    ::std::regex reg(R"(layout.*)");
    ::std::smatch m;
    auto pos = fullsource.cbegin();
    auto end = fullsource.cend();

    Type_STLVector<Type_ANSIString> layouts;
    for (; ::std::regex_search(pos, end, m, reg); pos = m.suffix().first) {
        layouts.emplace_back(m.str().c_str());
    }

    // match descriptor layouts
    GLSLReflect_DescriptorBindingName(layouts);

    // match push contants
    GLSLReflect_PushConstantName(layouts);

    // match outputs
    GLSLReflect_OutVarName(layouts);
}

void ShaderBase::GLSLReflect_DescriptorBindingName(
    Type_STLVector<Type_ANSIString> const& layouts) {
    ::std::regex reg(
        R"(.*(set)\s*=\s*([0-9])+,\s*(binding)\s*=\s*([0-9])+\s*(.*)\)\s*(buffer|uniform)\s*(image2D|sampler2D|uimage2D)*\s+(\w*)[;|{|\s]*)");
    ::std::smatch m;
    for (auto const& layout : layouts) {
        auto pos = layout.cbegin();
        auto end = layout.cend();

        if (::std::regex_match(pos, end, m, reg)) {
            GLSL_SetBindingInfo info;
            ::std::string name;
            info.set = atoi(m.str(2).c_str());
            info.binding = atoi(m.str(4).c_str());
            name = m.str(8);
            ::std::erase(name, ' ');

            ::std::regex postReg(R"((.*)\[([0-9]*)\])");
            if (::std::regex_match(name.cbegin(), name.cend(), m, postReg)) {
                name = m.str(1);
            }

            mGLSL_SetBindingNameMap.emplace_back(info, name.c_str());
        }
    }
}

void ShaderBase::GLSLReflect_PushConstantName(
    Type_STLVector<Type_ANSIString> const& layouts) {
    ::std::smatch m;
    ::std::regex pcReg(
        R"(.*\(\s*(push_constant)\s*\)\s+(uniform)\s+(.*)\s*\{*)");
    for (auto const& layout : layouts) {
        if (::std::regex_match(layout.cbegin(), layout.cend(), m, pcReg)) {
            auto name = m.str(3);
            ::std::erase(name, ' ');
            mGLSL_PushContantName = name.c_str();
            break;
        }
    }
}

void ShaderBase::GLSLReflect_OutVarName(
    Type_STLVector<Type_ANSIString> const& layouts) {
    Type_STLMap<uint32_t, Type_STLString> tempMap;

    ::std::smatch m;
    ::std::regex pcReg(
        R"(.*\(\s*(location)\s*=\s*([0-9])\s*\)\s+(out)\s+([^\s]*)\s*(.*)+\s*[;|{])");
    for (auto const& layout : layouts) {
        if (::std::regex_match(layout.cbegin(), layout.cend(), m, pcReg)) {
            uint32_t idx = atoi(m.str(2).c_str());
            auto name = m.str(5);
            ::std::erase(name, ' ');
            tempMap.emplace(idx, name.c_str());
        }
    }

    for (auto const& [_, name] : tempMap) {
        mGLSL_OutVarNames.push_back(name);
    }
}

Shader::Shader(VulkanContext& context, const char* name, const char* path,
               vk::ShaderStageFlagBits stage, const char* entry, void* pNext)
    : ShaderBase(context, name, path, stage, entry) {
    mHandle = CreateShader(pNext);
}

Shader::Shader(VulkanContext& context, const char* name, const char* sourcePath,
               vk::ShaderStageFlagBits stage, bool hasIncludes,
               Type_ShaderMacros const& defines, const char* entry, void* pNext)
    : ShaderBase(context, name, sourcePath, stage, hasIncludes, defines,
                 entry) {
    mHandle = CreateShader(pNext);
}

Shader::~Shader() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::PipelineShaderStageCreateInfo Shader::GetStageInfo(void* pNext) const {
    vk::PipelineShaderStageCreateInfo info;
    info.setModule(mHandle)
        .setStage(mStage)
        .setPName(mEntry.c_str())
        .setPNext(pNext);

    return info;
}

vk::ShaderModule Shader::GetHandle() const {
    return mHandle;
}

vk::ShaderModule Shader::CreateShader(void* pNext) const {
    vk::ShaderModuleCreateInfo createInfo {};
    createInfo.setCode(mSPIRVBinaryCode).setPNext(pNext);

    return mContext.GetDevice()->createShaderModule(createInfo);
}

ShaderObject::ShaderObject(VulkanContext& context, const char* name,
                           const char* spirvPath, vk::ShaderStageFlagBits stage,
                           vk::ShaderCreateFlagBitsEXT flags, const char* entry,
                           void* pNext)
    : ShaderBase(context, name, spirvPath, stage, entry) {
    mHandle = CreateShader(flags, pNext);
}

ShaderObject::ShaderObject(VulkanContext& context, const char* name,
                           const char* sourcePath,
                           vk::ShaderStageFlagBits stage, bool hasIncludes,
                           Type_ShaderMacros const& defines,
                           vk::ShaderCreateFlagBitsEXT flags, const char* entry,
                           void* pNext)
    : ShaderBase(context, name, sourcePath, stage, hasIncludes, defines,
                 entry) {
    mHandle = CreateShader(flags, pNext);
}

ShaderObject::~ShaderObject() {
    mContext.GetDevice()->destroy(mHandle);
}

vk::ShaderEXT ShaderObject::GetHandle() const {
    return mHandle;
}

Type_STLVector<vk::DescriptorSetLayout> ShaderObject::GetDescLayoutHandles()
    const {
    Type_STLVector<vk::DescriptorSetLayout> layouts {};
    layouts.reserve(mDescLayouts.size());
    for (auto const& layout : mDescLayouts) {
        layouts.push_back(layout->GetHandle());
    }
    return layouts;
}

vk::ShaderEXT ShaderObject::CreateShader(vk::ShaderCreateFlagBitsEXT flags,
                                         void* pNext) {
    const auto datas =
        MergeDescLayoutData(Type_STLVector<ShaderBase*> {(ShaderBase*)this});

    mDescLayouts = CreateDescLayout(mContext, datas, nullptr);

    auto layouts = GetDescLayoutHandles();

    auto pushConstant = GetPushContantData();

    vk::ShaderStageFlagBits nextStage {};
    if (mStage == vk::ShaderStageFlagBits::eTaskEXT) {
        nextStage = vk::ShaderStageFlagBits::eMeshEXT;
    } else if (mStage == vk::ShaderStageFlagBits::eMeshEXT) {
        nextStage = vk::ShaderStageFlagBits::eFragment;
    }

    vk::ShaderCreateInfoEXT shaderCreateInfo {};
    shaderCreateInfo.setFlags(flags)
        .setStage(mStage)
        .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
        .setPCode(mSPIRVBinaryCode.data())
        .setCodeSize(sizeof(mSPIRVBinaryCode[0]) * mSPIRVBinaryCode.size())
        .setNextStage(nextStage)
        .setPName("main")
        .setPNext(pNext);

    if (!layouts.empty())
        shaderCreateInfo.setSetLayouts(layouts);

    if (pushConstant)
        shaderCreateInfo.setPushConstantRanges(pushConstant.value());

    const auto res = mContext.GetDevice()->createShaderEXT(shaderCreateInfo);

    VK_CHECK(res.result);

    return res.value;
}

ShaderProgram::ShaderProgram(Shader* comp, void* layoutPNext)
    : mContext(comp->mContext) {
    SetShader(ShaderStage::Compute, comp);
    GenerateProgram(layoutPNext);
}

ShaderProgram::ShaderProgram(Shader* vert, Shader* frag, void* layoutPNext)
    : mContext(vert->mContext) {
    SetShader(ShaderStage::Vertex, vert);
    SetShader(ShaderStage::Fragment, frag);
    GenerateProgram(layoutPNext);
}

ShaderProgram::ShaderProgram(Shader* task, Shader* mesh, Shader* frag,
                             void* layoutPNext)
    : mContext(mesh->mContext) {
    if (task)
        SetShader(ShaderStage::Task, task);
    SetShader(ShaderStage::Mesh, mesh);
    SetShader(ShaderStage::Fragment, frag);
    GenerateProgram(layoutPNext);
}

ShaderProgram::ShaderProgram(ShaderObject* comp, void* layoutPNext)
    : mContext(comp->mContext) {
    SetShader(ShaderStage::Compute, comp);

    if (auto pcData = comp->GetPushContantData()) {
        mCombinedPushContants.emplace(comp->mGLSL_PushContantName,
                                      pcData.value());
    }

    mDescLayouts = comp->mDescLayouts;
}

ShaderProgram::ShaderProgram(ShaderObject* task, ShaderObject* mesh,
                             ShaderObject* frag, void* layoutPNext)
    : mContext(mesh->mContext) {
    if (task)
        SetShader(ShaderStage::Task, task);
    SetShader(ShaderStage::Mesh, mesh);
    SetShader(ShaderStage::Fragment, frag);

    for (auto pShader : pShaders) {
        if (pShader) {
            if (auto pcData = pShader->GetPushContantData()) {
                mCombinedPushContants.emplace(pShader->mGLSL_PushContantName,
                                              pcData.value());
                break;
            }
        }
    }

    MergeDescLayoutDatas(layoutPNext);

    mRtvNames = frag->mGLSL_OutVarNames;
}

const ShaderBase* ShaderProgram::operator[](ShaderStage stage) const {
    return pShaders[Utils::EnumCast(stage)];
}

ShaderBase* ShaderProgram::operator[](ShaderStage stage) {
    return pShaders[Utils::EnumCast(stage)];
}

::std::array<ShaderBase*, Utils::EnumCast(ShaderStage::Count)>
ShaderProgram::GetShaderArray() const {
    return pShaders;
}

Type_STLVector<vk::PushConstantRange> ShaderProgram::GetPCRanges() const {
    Type_STLVector<vk::PushConstantRange> temp;
    if (mCombinedPushContants) {
        temp.push_back(mCombinedPushContants->second);
    }
    return temp;
}

ShaderProgram::Type_CombinedPushContant const&
ShaderProgram::GetCombinedPushContant() const {
    return mCombinedPushContants;
}

Type_STLVector<DescriptorSetLayout*> ShaderProgram::GetCombinedDescLayouts()
    const {
    Type_STLVector<DescriptorSetLayout*> layouts {};
    layouts.reserve(mDescLayouts.size());
    for (auto const& layout : mDescLayouts) {
        layouts.push_back(layout.get());
    }
    return layouts;
}

Type_STLVector<vk::DescriptorSetLayout>
ShaderProgram::GetCombinedDescLayoutHandles() const {
    Type_STLVector<vk::DescriptorSetLayout> layouts {};
    layouts.reserve(mDescLayouts.size());
    for (auto const& layout : mDescLayouts) {
        layouts.push_back(layout->GetHandle());
    }
    return layouts;
}

Type_STLVector<Type_STLString> const& ShaderProgram::GetRTVNames() const {
    return mRtvNames;
}

void ShaderProgram::SetShader(ShaderStage stage, ShaderBase* shader) {
    pShaders[Utils::EnumCast(stage)] = shader;
}

void ShaderProgram::GenerateProgram(void* layoutPNext) {
    MergeDescLayoutDatas(layoutPNext);
    MergePushContantDatas();

    if (auto frag = pShaders[Utils::EnumCast(ShaderStage::Fragment)]) {
        mRtvNames = frag->mGLSL_OutVarNames;
    }
}

void ShaderProgram::MergeDescLayoutDatas(void* pNext) {
    Type_STLVector<ShaderBase::DescriptorSetLayoutData> datas {};

    Type_STLVector<ShaderBase*> shaders {};
    for (auto const& shader : pShaders) {
        if (shader)
            shaders.push_back(shader);
    }

    datas = MergeDescLayoutData(shaders);

    CreateDescLayouts(datas, pNext);
}

void ShaderProgram::MergePushContantDatas() {
    for (auto const& shader : pShaders)
        if (shader) {
            if (auto data = shader->mPushContantData) {
                if (mCombinedPushContants) {
                    auto& [name, range] = *mCombinedPushContants;
                    if (range.size != data->size
                        || range.offset != data->offset) {
                        throw ::std::runtime_error(
                            "all stages in pipeline share one unique push "
                            "constant");
                    } else {
                        range.stageFlags |= data->stageFlags;
                    }
                } else {
                    mCombinedPushContants = ::std::make_optional<
                        ::std::pair<Type_STLString, vk::PushConstantRange>>();
                    mCombinedPushContants.emplace(
                        shader->mGLSL_PushContantName.c_str(), data.value());
                }
            }
        }
}

void ShaderProgram::CreateDescLayouts(
    Type_STLVector<Shader::DescriptorSetLayoutData> const& datas, void* pNext) {
    mDescLayouts = CreateDescLayout(mContext, datas, pNext);
}

}  // namespace IntelliDesign_NS::Vulkan::Core