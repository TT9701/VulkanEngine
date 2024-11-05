#pragma once

#include <mutex>

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Native/Descriptors.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

enum class ShaderStage {
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
    Task = 3,
    Mesh = 4,
    Count
};

using Type_ShaderMacros = Type_STLUnorderedMap_String<Type_STLString>;

class ShaderProgram;

// TODO: Specialization parameters
class Shader {
    struct DescriptorSetLayoutData {
        Type_STLString name;
        uint32_t setIdx;
        uint32_t bindingIdx;
        vk::DescriptorType type;
        vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eAll;
        uint32_t descCount = 1;
    };

    struct GLSL_SetBindingInfo {
        uint32_t set {0};
        uint32_t binding {0};
    };

    using Type_GLSL_SetBindingName_Map =
        Type_STLVector<::std::pair<GLSL_SetBindingInfo, Type_STLString>>;

public:
    friend class ShaderProgram;

    // from spirv
    Shader(Context* context, const char* name, const char* spirvPath,
           vk::ShaderStageFlagBits stage, const char* entry = "main",
           void* pNext = nullptr);

    // from glsl source code
    Shader(Context* context, const char* name, const char* sourcePath,
           vk::ShaderStageFlagBits stage, bool hasIncludes,
           Type_ShaderMacros const& defines, const char* entry = "main",
           void* pNext = nullptr);

    ~Shader();
    MOVABLE_ONLY(Shader);

public:
    Type_STLVector<DescriptorSetLayoutData> SPIRVReflect_DescSetLayouts();

    ::std::optional<vk::PushConstantRange> SPIRVReflect_PushContants();

    vk::ShaderModule GetHandle() const { return mShader; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

    ::std::span<uint32_t> GetBinaryCode();

    ::std::span<DescriptorSetLayoutData> GetDescSetLayoutDatas();

    ::std::optional<vk::PushConstantRange> const& GetPushContantData() const;

    Type_STLString const& GetName() const;

    ::std::mutex& GetMutex();

private:
    vk::ShaderModule CreateShader(void* pNext) const;

    void GLSLReflect(Type_STLString const& source);
    void GLSLReflect_DescriptorBindingName(
        Type_STLVector<Type_STLString> const& layouts);
    void GLSLReflect_PushConstantName(
        Type_STLVector<Type_STLString> const& layouts);
    void GLSLReflect_OutVarName(Type_STLVector<Type_STLString> const& layouts);

private:
    Context* pContext;
    Type_STLString mName;

    Type_STLString mEntry;
    vk::ShaderStageFlagBits mStage;

    Type_STLVector<uint32_t> mSPIRVBinaryCode {};

    Type_GLSL_SetBindingName_Map mGLSL_SetBindingNameMap {};
    Type_STLString mGLSL_PushContantName {};
    Type_STLVector<Type_STLString> mGLSL_OutVarNames {};

    vk::ShaderModule mShader {};

    Type_STLVector<DescriptorSetLayoutData> mDescSetLayoutDatas {};

    ::std::optional<vk::PushConstantRange> mPushContantData {};

    ::std::mutex mMutex {};
};

class ShaderProgram {
    friend class PipelineLayout;

public:
    using Type_CombinedPushContant =
        Type_STLVector<::std::pair<Type_STLString, vk::PushConstantRange>>;
    using Type_ShaderArray =
        ::std::array<Shader*, Utils::EnumCast(ShaderStage::Count)>;

public:
    ShaderProgram(Shader* comp, void* layoutPNext = nullptr);
    ShaderProgram(Shader* vert, Shader* frag, void* layoutPNext = nullptr);
    ShaderProgram(Shader* task, Shader* mesh, Shader* frag,
                  void* layoutPNext = nullptr);

    ~ShaderProgram() = default;
    MOVABLE_ONLY(ShaderProgram);

    const Shader* operator[](ShaderStage stage) const;
    Shader* operator[](ShaderStage stage);

    Type_ShaderArray GetShaderArray() const;

    Type_STLVector<vk::PushConstantRange> GetPCRanges() const;
    Type_CombinedPushContant const& GetCombinedPushContant() const;

    Type_STLVector<DescriptorSetLayout*> GetCombinedDescLayouts() const;
    Type_STLVector<vk::DescriptorSetLayout> GetCombinedDescLayoutHandles()
        const;

private:
    void SetShader(ShaderStage stage, Shader* shader);
    void GenerateProgram(void* layoutPNext);

    void MergeDescLayoutDatas(void* pNext);
    void MergePushContantDatas();
    void CreateDescLayouts(
        Type_STLVector<Shader::DescriptorSetLayoutData> const& datas,
        void* pNext);

private:
    Context* pContext;
    Type_ShaderArray pShaders {nullptr};

    Type_CombinedPushContant mCombinedPushContants {};

    Type_STLVector<SharedPtr<DescriptorSetLayout>> mDescLayouts {};

    Type_STLVector<Type_STLString> mRtvNames {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core