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
    Type_STLVector<DescriptorSetLayoutData> ReflectDescSetLayouts();

    ::std::optional<vk::PushConstantRange> ReflectPushContants();

    vk::ShaderModule GetHandle() const { return mShader; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

    ::std::span<uint32_t> GetBinaryCode();

    ::std::span<DescriptorSetLayoutData> GetDescSetLayoutDatas();

    ::std::optional<vk::PushConstantRange> const& GetPushContantData() const;

    Type_STLString const& GetName() const;

    ::std::mutex& GetMutex();

public:
    vk::ShaderModule CreateShader(void* pNext) const;

private:
    Context* pContext;
    Type_STLString mName;

    Type_STLString mEntry;
    vk::ShaderStageFlagBits mStage;

    Type_STLVector<uint32_t> mSPIRVBinaryCode {};

    vk::ShaderModule mShader {};

    Type_STLVector<DescriptorSetLayoutData> mDescSetLayoutDatas {};

    ::std::optional<vk::PushConstantRange> mPushContantData {};

    ::std::mutex mMutex {};
};

class ShaderProgram {
public:
    ShaderProgram(Shader* comp, void* layoutPNext = nullptr);
    ShaderProgram(Shader* vert, Shader* frag, void* layoutPNext = nullptr);
    ShaderProgram(Shader* task, Shader* mesh, Shader* frag,
                  void* layoutPNext = nullptr);

    ~ShaderProgram() = default;
    MOVABLE_ONLY(ShaderProgram);

    const Shader* operator[](ShaderStage stage) const;
    Shader* operator[](ShaderStage stage);

    ::std::array<Shader*, Utils::EnumCast(ShaderStage::Count)> GetShaderArray()
        const;

    Type_STLVector<vk::PushConstantRange> const& GetCombinedPushConstants()
        const;

    Type_STLVector<DescriptorSetLayout*> GetCombinedDescLayouts() const;
    Type_STLVector<vk::DescriptorSetLayout> GetCombinedDescLayoutHandles()
        const;

private:
    void SetShader(ShaderStage stage, Shader* shader);
    void GenerateProgram(void* layoutPNext);

    void MergeDescLayoutDatas(void* pNext);
    void MergePushContantDatas();
    void CreateDescLayouts(Type_STLVector<Shader::DescriptorSetLayoutData> const& datas, void* pNext);

private:
    Context* pContext;
    ::std::array<Shader*, Utils::EnumCast(ShaderStage::Count)> pShaders {
        nullptr};

    Type_STLVector<vk::PushConstantRange> mCombinedPushContants {};

    Type_STLVector<SharedPtr<DescriptorSetLayout>> mDescLayouts {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core