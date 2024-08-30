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
class ShaderModule {
public:
    // from spirv
    ShaderModule(Context* context, const char* spirvPath,
                 ShaderStage stage, const char* entry = "main",
                 void* pNext = nullptr);

    // from glsl source code
    ShaderModule(Context* context, const char* sourcePath,
                 ShaderStage stage, bool hasIncludes,
                 Type_ShaderMacros const& defines, const char* entry = "main",
                 void* pNext = nullptr);

    ~ShaderModule();
    MOVABLE_ONLY(ShaderModule);

public:
    vk::ShaderModule GetHandle() const { return mShaderModule; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

public:
    vk::ShaderModule CreateShaderModule(::std::span<uint32_t> binaryCode,
                                        void* pNext) const;

private:
    Context* pContext;

    Type_STLString mEntry;
    ShaderStage mStage;

    vk::ShaderModule mShaderModule {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core