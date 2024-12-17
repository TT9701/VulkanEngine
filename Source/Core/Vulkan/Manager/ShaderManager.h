#pragma once

#include <mutex>

#include "Core/Vulkan/Native/Shader.h"

/*
 * Shader name : "name_glsl460_stage_(macros)_entrypoint"
 */

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;

class ShaderManager {
    using Type_Shaders = Type_STLUnorderedMap_String<SharedPtr<Shader>>;
    using Type_Programs = Type_STLUnorderedMap_String<SharedPtr<ShaderProgram>>;

public:
    ShaderManager(VulkanContext& context);

    // persist shader module from spir-v binary code
    SharedPtr<Shader> CreateShaderFromSPIRV(const char* name,
                                            const char* spirvPath,
                                            vk::ShaderStageFlagBits stage,
                                            const char* entry = "main",
                                            void* pNext = nullptr);

    // persist shader module from glsl source code
    SharedPtr<Shader> CreateShaderFromGLSL(
        const char* name, const char* sourcePath, vk::ShaderStageFlagBits stage,
        bool hasIncludes = false, Type_ShaderMacros const& defines = {},
        const char* entry = "main", void* pNext = nullptr);

    void ReleaseShader(const char* name, vk::ShaderStageFlagBits stage,
                       Type_ShaderMacros const& defines = {},
                       const char* entry = "main");

    Shader* GetShader(const char* name, vk::ShaderStageFlagBits stage,
                      Type_ShaderMacros const& defines = {},
                      const char* entry = "main");

    Type_STLString ParseShaderName(const char* name,
                                   vk::ShaderStageFlagBits stage,
                                   Type_ShaderMacros const& defines = {},
                                   const char* entry = "main") const;

    ShaderProgram* CreateProgram(const char* name, Shader* comp);
    ShaderProgram* CreateProgram(const char* name, Shader* vert, Shader* frag);
    ShaderProgram* CreateProgram(const char* name, Shader* task, Shader* mesh,
                                 Shader* frag);

    ShaderProgram* GetProgram(const char* name) const;

private:
    VulkanContext& mContext;

    Type_Shaders mShaders {};
    Type_Programs mPrograms {};
    ::std::mutex mMutex;
};

}  // namespace IntelliDesign_NS::Vulkan::Core