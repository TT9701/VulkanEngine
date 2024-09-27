#include "ShaderManager.hpp"

#include "Context.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

ShaderManager::ShaderManager(Context* context) : pContext(context) {}

SharedPtr<Shader> ShaderManager::CreateShaderFromSPIRV(
    const char* name, const char* spirvPath, vk::ShaderStageFlagBits stage,
    const char* entry, void* pNext) {
    auto shaderName = ParseShaderName(name, stage, {}, entry);
    auto ptr = MakeShared<Shader>(pContext, shaderName.c_str(), spirvPath,
                                  stage, entry, pNext);
    pContext->SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaders.emplace(shaderName, ptr);
    return ptr;
}

SharedPtr<Shader> ShaderManager::CreateShaderFromSource(
    const char* name, const char* sourcePath, vk::ShaderStageFlagBits stage,
    bool hasIncludes, Type_ShaderMacros const& defines, const char* entry,
    void* pNext) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    auto ptr = MakeShared<Shader>(pContext, shaderName.c_str(), sourcePath,
                                  stage, hasIncludes, defines, entry, pNext);
    pContext->SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaders.emplace(shaderName, ptr);
    return ptr;
}

void ShaderManager::ReleaseShader(const char* name,
                                  vk::ShaderStageFlagBits stage,
                                  Type_ShaderMacros const& defines,
                                  const char* entry) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    ::std::unique_lock<::std::mutex> lock {mMutex};
    auto it = mShaders.find(shaderName);
    if (it != mShaders.end()) {
        ::std::unique_lock<::std::mutex> ll {it->second->GetMutex()};
        mShaders.erase(it);
    }
}

Shader* ShaderManager::GetShader(const char* name,
                                 vk::ShaderStageFlagBits stage,
                                 Type_ShaderMacros const& defines,
                                 const char* entry) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    ::std::unique_lock<::std::mutex> lock {mMutex};
    return mShaders.at(shaderName).get();
}

struct Comp {
    template <typename T>
    bool operator()(const T& l, const T& r) const {
        if (l.first != r.first) {
            return l.first < r.first;
        }
        return l.second < r.second;
    }
};

Type_STLString ShaderManager::ParseShaderName(const char* name,
                                              vk::ShaderStageFlagBits stage,
                                              Type_ShaderMacros const& defines,
                                              const char* entry) const {
    Type_STLString res {};
    res = name;

    // TODO: hlsl
    res.append("@glsl460").append(vk::to_string(stage).insert(0, "@"));

    if (!defines.empty()) {
        ::std::set<::std::pair<Type_STLString, Type_STLString>, Comp> temp {
            defines.begin(), defines.end()};
        for (auto const& [macro, value] : temp) {
            res.append("@" + macro + "@" + value);
        }
    }

    res.append(Type_STLString("@") + entry);

    return res;
}

}  // namespace IntelliDesign_NS::Vulkan::Core