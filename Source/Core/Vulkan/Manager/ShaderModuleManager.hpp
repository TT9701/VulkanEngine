#pragma once

#include "Core/Vulkan/Native/ShaderModule.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

class Context;

class ShaderModuleManager {
    using Type_ShaderModules =
        Type_STLUnorderedMap_String<SharedPtr<ShaderModule>>;

public:
    ShaderModuleManager(Context* context);

    // temporary shader module from spir-v binary code
    SharedPtr<ShaderModule> CreateShaderModule(const char* name,
                                               const char* spirvPath,
                                               ShaderStage stage,
                                               const char* entry = "main",
                                               void* pNext = nullptr);

    // temporary shader module from glsl source code
    SharedPtr<ShaderModule> CreateShaderModule(
        const char* name, const char* sourcePath, ShaderStage stage,
        bool hasIncludes, Type_ShaderMacros const& defines,
        const char* entry = "main", void* pNext = nullptr);

    // persist shader module from spir-v binary code
    SharedPtr<ShaderModule> CreatePersistShaderModule(
        const char* name, const char* spirvPath, ShaderStage stage,
        const char* entry = "main", void* pNext = nullptr);

    // persist shader module from glsl source code
    SharedPtr<ShaderModule> CreatePersistShaderModule(
        const char* name, const char* sourcePath, ShaderStage stage,
        bool hasIncludes, Type_ShaderMacros const& defines,
        const char* entry = "main", void* pNext = nullptr);

private:
    Context* pContext;

    // only shaders which would be used multiple times (created with CreatePersistShaderModule()), store in this map.
    Type_ShaderModules mShaderModules {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core