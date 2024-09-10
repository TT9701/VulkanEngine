#pragma once

#include <mutex>

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"
#include "Core/Vulkan/Manager/DescriptorManager.hpp"
#include "Core/Vulkan/Native/Descriptors.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

using Type_ShaderMacros = Type_STLUnorderedMap_String<Type_STLString>;

// TODO: Specialization parameters
class Shader {
public:
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
    void ReflectDescSetLayouts();

    vk::ShaderModule GetHandle() const { return mShader; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

    ::std::span<uint32_t> GetBinaryCode();

    ::std::span<DescriptorSetLayoutData> GetDescSetLayoutDatas();

    ::std::optional<vk::PushConstantRange> const& GetPushContantData() const;

    Type_STLString const& GetName() const;

    Type_STLVector<vk::DescriptorSetLayout>& GetAllDescSetLayoutHandles();

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

    Type_STLVector<vk::DescriptorSetLayout> mDescSetLayouts {};

    Type_STLVector<DescriptorSetLayoutData> mDescSetLayoutDatas {};
    ::std::optional<vk::PushConstantRange> mPushContantData {};

    ::std::mutex mMutex {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core