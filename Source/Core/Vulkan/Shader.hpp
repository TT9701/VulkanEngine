#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

enum class ShaderStage { Compute, Vertex, Fragment };

class Context;

// TODO: Specialization parameters
class Shader {
public:
    Shader(Context* context, ::std::string name,
                 ::std::vector<uint32_t> const& binaryCode, ShaderStage stage,
                 ::std::string entry = "main", void* pNext = nullptr);

    Shader(Context* context, ::std::string const& name,
                 ::std::string const& path, ShaderStage stage,
                 ::std::string const& entry = "main", void* pNext = nullptr);

    ~Shader();
    MOVABLE_ONLY(Shader);

public:
    vk::ShaderModule GetHandle() const { return mShaderModule; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

public:
    vk::ShaderModule CreateShaderModule(
        ::std::vector<uint32_t> const& binaryCode, void* pNext) const;

private:
    Context* pContext;

    ::std::string mName;
    ::std::string mEntry;
    ShaderStage mStage;

    vk::ShaderModule mShaderModule;
};

}  // namespace IntelliDesign_NS::Vulkan::Core