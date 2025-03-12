#pragma once

#include <mutex>
#include <optional>

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.h"
#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Descriptors.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

enum class ShaderStage { Vertex, Fragment, Compute, Task, Mesh, Count };

using Type_ShaderMacros = Type_STLUnorderedMap_String<Type_STLString>;

class ShaderProgram;

struct ShaderIDInfo {
    const char* name;
    vk::ShaderStageFlagBits stage;
    Type_ShaderMacros macros {};
    const char* entry {"main"};
};

class ShaderBase {
public:
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
    // from spirv
    ShaderBase(VulkanContext& context, const char* name, const char* spirvPath,
               vk::ShaderStageFlagBits stage, const char* entry = "main");

    // from glsl source code
    ShaderBase(VulkanContext& context, const char* name, const char* sourcePath,
               vk::ShaderStageFlagBits stage, bool hasIncludes,
               Type_ShaderMacros const& defines, const char* entry = "main");

    CLASS_NO_COPY_MOVE(ShaderBase);

    Type_STLVector<DescriptorSetLayoutData> SPIRVReflect_DescSetLayouts();

    ::std::optional<vk::PushConstantRange> SPIRVReflect_PushContants();

    ::std::span<uint32_t> GetBinaryCode();

    ::std::span<DescriptorSetLayoutData> GetDescSetLayoutDatas();

    ::std::optional<vk::PushConstantRange> const& GetPushContantData() const;

    Type_STLString const& GetName() const;

    ::std::mutex& GetMutex();

protected:
    friend class ShaderProgram;

    void GLSLReflect(Type_ANSIString const& source);
    void GLSLReflect_DescriptorBindingName(
        Type_STLVector<Type_ANSIString> const& layouts);
    void GLSLReflect_PushConstantName(
        Type_STLVector<Type_ANSIString> const& layouts);
    void GLSLReflect_OutVarName(Type_STLVector<Type_ANSIString> const& layouts);

protected:
    VulkanContext& mContext;
    Type_STLString mName;

    Type_STLString mEntry;
    vk::ShaderStageFlagBits mStage;

    Type_STLVector<uint32_t> mSPIRVBinaryCode {};

    Type_GLSL_SetBindingName_Map mGLSL_SetBindingNameMap {};
    Type_STLString mGLSL_PushContantName {};
    Type_STLVector<Type_STLString> mGLSL_OutVarNames {};

    Type_STLVector<DescriptorSetLayoutData> mDescSetLayoutDatas {};

    ::std::optional<vk::PushConstantRange> mPushContantData {};

    ::std::mutex mMutex {};
};

// TODO: Specialization parameters
class Shader : public ShaderBase {
public:
    // from spirv
    Shader(VulkanContext& context, const char* name, const char* spirvPath,
           vk::ShaderStageFlagBits stage, const char* entry = "main",
           void* pNext = nullptr);

    // from glsl source code
    Shader(VulkanContext& context, const char* name, const char* sourcePath,
           vk::ShaderStageFlagBits stage, bool hasIncludes,
           Type_ShaderMacros const& defines, const char* entry = "main",
           void* pNext = nullptr);

    ~Shader();
    CLASS_NO_COPY_MOVE(Shader);

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

    vk::ShaderModule GetHandle() const;

private:
    vk::ShaderModule CreateShader(void* pNext) const;

    vk::ShaderModule mHandle {};
};

class ShaderObject : public ShaderBase {
    friend ShaderProgram;

public:
    // from spirv
    ShaderObject(VulkanContext& context, const char* name,
                 const char* spirvPath, vk::ShaderStageFlagBits stage,
                 vk::ShaderCreateFlagBitsEXT flags =
                     vk::ShaderCreateFlagBitsEXT::eIndirectBindable,
                 const char* entry = "main", void* pNext = nullptr);

    // from glsl source code
    ShaderObject(VulkanContext& context, const char* name,
                 const char* sourcePath, vk::ShaderStageFlagBits stage,
                 bool hasIncludes, Type_ShaderMacros const& defines,
                 vk::ShaderCreateFlagBitsEXT flags =
                     vk::ShaderCreateFlagBitsEXT::eIndirectBindable,
                 const char* entry = "main", void* pNext = nullptr);

    ~ShaderObject();
    CLASS_NO_COPY_MOVE(ShaderObject);

    vk::ShaderEXT GetHandle() const;

    Type_STLVector<vk::DescriptorSetLayout> GetDescLayoutHandles() const;

private:
    vk::ShaderEXT CreateShader(vk::ShaderCreateFlagBitsEXT flags, void* pNext);

    Type_STLVector<SharedPtr<DescriptorSetLayout>> mDescLayouts {};

    vk::ShaderEXT mHandle {};
};

class ShaderProgram {
    friend class PipelineLayout;

public:
    using Type_CombinedPushContant =
        ::std::optional<::std::pair<Type_STLString, vk::PushConstantRange>>;
    using Type_ShaderArray =
        ::std::array<ShaderBase*, Utils::EnumCast(ShaderStage::Count)>;

public:
    ShaderProgram(Shader* comp, void* layoutPNext = nullptr);
    ShaderProgram(Shader* vert, Shader* frag, void* layoutPNext = nullptr);
    ShaderProgram(Shader* task, Shader* mesh, Shader* frag,
                  void* layoutPNext = nullptr);

    ShaderProgram(ShaderObject* comp, void* layoutPNext = nullptr);
    ShaderProgram(ShaderObject* task, ShaderObject* mesh, ShaderObject* frag,
                  void* layoutPNext = nullptr);

    ~ShaderProgram() = default;
    CLASS_MOVABLE_ONLY(ShaderProgram);

    const ShaderBase* operator[](ShaderStage stage) const;
    ShaderBase* operator[](ShaderStage stage);

    Type_ShaderArray GetShaderArray() const;

    Type_STLVector<vk::PushConstantRange> GetPCRanges() const;
    Type_CombinedPushContant const& GetCombinedPushContant() const;

    Type_STLVector<DescriptorSetLayout*> GetCombinedDescLayouts() const;
    Type_STLVector<vk::DescriptorSetLayout> GetCombinedDescLayoutHandles()
        const;

    Type_STLVector<Type_STLString> const& GetRTVNames() const;

private:
    void SetShader(ShaderStage stage, ShaderBase* shader);
    void GenerateProgram(void* layoutPNext);

    void MergeDescLayoutDatas(void* pNext);
    void MergePushContantDatas();
    void CreateDescLayouts(
        Type_STLVector<Shader::DescriptorSetLayoutData> const& datas,
        void* pNext);

private:
    VulkanContext& mContext;
    Type_ShaderArray pShaders {nullptr};

    Type_CombinedPushContant mCombinedPushContants {};

    Type_STLVector<SharedPtr<DescriptorSetLayout>> mDescLayouts {};

    Type_STLVector<Type_STLString> mRtvNames {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core