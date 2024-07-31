#pragma once

#include <vulkan/vulkan.hpp>

#include "Core/Utilities/Defines.hpp"

enum class ShaderStage { Compute, Vertex, Fragment };

class VulkanContext;

// TODO: Specialization parameters
class VulkanShader {
public:
    VulkanShader(VulkanContext* context, ::std::string name,
                 ::std::vector<uint32_t> const& binaryCode, ShaderStage stage,
                 ::std::string entry = "main", void* pNext = nullptr);

    VulkanShader(VulkanContext* context, ::std::string const& name,
                 ::std::string const& path, ShaderStage stage,
                 ::std::string const& entry = "main", void* pNext = nullptr);

    ~VulkanShader();
    MOVABLE_ONLY(VulkanShader);

public:
    vk::ShaderModule GetHandle() const { return mShaderModule; }

    vk::PipelineShaderStageCreateInfo GetStageInfo(void* pNext = nullptr) const;

public:
    vk::ShaderModule CreateShaderModule(
        ::std::vector<uint32_t> const& binaryCode, void* pNext) const;

private:
    VulkanContext* pContext;

    ::std::string mName;
    ::std::string mEntry;
    ShaderStage   mStage;

    vk::ShaderModule mShaderModule;
};