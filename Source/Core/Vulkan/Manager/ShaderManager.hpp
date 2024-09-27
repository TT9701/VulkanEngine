#pragma once

#include <mutex>

#include "Core/Vulkan/Native/Shader.hpp"

/*
 * Shader name : "name_glsl460_stage_(macros)_entrypoint"
 */

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class ShaderManager {
    using Type_Shaders = Type_STLUnorderedMap_String<SharedPtr<Shader>>;

public:
    ShaderManager(Context* context);

    // persist shader module from spir-v binary code
    SharedPtr<Shader> CreateShaderFromSPIRV(const char* name,
                                            const char* spirvPath,
                                            vk::ShaderStageFlagBits stage,
                                            const char* entry = "main",
                                            void* pNext = nullptr);

    // persist shader module from glsl source code
    SharedPtr<Shader> CreateShaderFromSource(
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

private:
    Context* pContext;

    Type_Shaders mShaders {};
    ::std::mutex mMutex;
};

}  // namespace IntelliDesign_NS::Vulkan::Core