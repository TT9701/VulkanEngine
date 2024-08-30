#pragma once

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"
#include "Core/Utilities/MemoryPool.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

enum class ShaderStage { Compute, Vertex, Fragment, Task, Mesh };

class Context;

using Type_ShaderMacros = Type_STLUnorderedMap_String<Type_STLString>;

// TODO: Specialization parameters
class Shader {
public:
    // from spirv
    Shader(Context* context, const char* name, const char* spirvPath,
           ShaderStage stage, const char* entry = "main",
           void* pNext = nullptr);

    // from glsl source code
    Shader(Context* context, const char* name, const char* sourcePath,
           ShaderStage stage, bool hasIncludes,
           Type_ShaderMacros const& defines, const char* entry = "main",
           void* pNext = nullptr);

    ~Shader();
    MOVABLE_ONLY(Shader);

public:
    vk::ShaderModule GetHandle() const { return mShaderModule; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

public:
    vk::ShaderModule CreateShaderModule(::std::span<uint32_t> binaryCode,
                                        void* pNext) const;

private:
    Context* pContext;

    Type_STLString mName;
    Type_STLString mEntry;
    ShaderStage mStage;

    vk::ShaderModule mShaderModule {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core