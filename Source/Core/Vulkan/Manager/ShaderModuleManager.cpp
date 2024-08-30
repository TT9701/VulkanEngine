#include "ShaderModuleManager.hpp"

#include "Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

ShaderModuleManager::ShaderModuleManager(Context* context)
    : pContext(context) {}

SharedPtr<ShaderModule> ShaderModuleManager::CreateShaderModule(
    const char* name, const char* spirvPath, ShaderStage stage,
    const char* entry, void* pNext) {
    auto ptr = MakeShared<ShaderModule>(pContext, spirvPath, stage, entry,
                                    pNext);
    pContext->SetName(ptr->GetHandle(), name);
    return ptr;
}

SharedPtr<ShaderModule> ShaderModuleManager::CreateShaderModule(
    const char* name, const char* sourcePath, ShaderStage stage,
    bool hasIncludes, Type_ShaderMacros const& defines, const char* entry,
    void* pNext) {
    auto ptr = MakeShared<ShaderModule>(pContext, sourcePath, stage,
                                    hasIncludes, defines, entry, pNext);
    pContext->SetName(ptr->GetHandle(), name);
    return ptr;
}

SharedPtr<ShaderModule> ShaderModuleManager::CreatePersistShaderModule(
    const char* name, const char* spirvPath, ShaderStage stage,
    const char* entry, void* pNext) {
    auto ptr = CreateShaderModule(name, spirvPath, stage, entry, pNext);
    mShaderModules.emplace(name, ptr);
    return ptr;
}

SharedPtr<ShaderModule> ShaderModuleManager::CreatePersistShaderModule(
    const char* name, const char* sourcePath, ShaderStage stage,
    bool hasIncludes, Type_ShaderMacros const& defines, const char* entry,
    void* pNext) {
    auto ptr = CreateShaderModule(name, sourcePath, stage, hasIncludes, defines,
                                  entry, pNext);
    mShaderModules.emplace(name, ptr);
    return ptr;
}

}  // namespace IntelliDesign_NS::Vulkan::Core