#include "ShaderManager.h"

#include "VulkanContext.h"

namespace IntelliDesign_NS::Vulkan::Core {

ShaderManager::ShaderManager(VulkanContext& context) : mContext(context) {}

SharedPtr<Shader> ShaderManager::CreateShaderFromSPIRV(
    const char* name, const char* spirvPath, vk::ShaderStageFlagBits stage,
    const char* entry, void* pNext) {
    auto shaderName = ParseShaderName(name, stage, {}, entry);
    auto ptr = MakeShared<Shader>(mContext, shaderName.c_str(), spirvPath,
                                  stage, entry, pNext);
    mContext.SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaders.emplace(shaderName, ptr);
    return ptr;
}

SharedPtr<Shader> ShaderManager::CreateShaderFromGLSL(
    const char* name, const char* sourcePath, vk::ShaderStageFlagBits stage,
    bool hasIncludes, Type_ShaderMacros const& defines, const char* entry,
    void* pNext) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    auto ptr = MakeShared<Shader>(mContext, shaderName.c_str(), sourcePath,
                                  stage, hasIncludes, defines, entry, pNext);
    mContext.SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaders.emplace(shaderName, ptr);
    return ptr;
}

SharedPtr<ShaderObject> ShaderManager::CreateShaderObjectFromSPIRV(
    const char* name, const char* spirvPath, vk::ShaderStageFlagBits stage,
    vk::ShaderCreateFlagBitsEXT flags, const char* entry, void* pNext) {
    auto shaderName = ParseShaderName(name, stage, {}, entry);
    auto ptr = MakeShared<ShaderObject>(mContext, shaderName.c_str(), spirvPath,
                                        stage, flags, entry, pNext);
    mContext.SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaderObjects.emplace(shaderName, ptr);
    return ptr;
}

SharedPtr<ShaderObject> ShaderManager::CreateShaderObjectFromGLSL(
    const char* name, const char* sourcePath, vk::ShaderStageFlagBits stage,
    vk::ShaderCreateFlagBitsEXT flags, bool hasIncludes,
    Type_ShaderMacros const& defines, const char* entry, void* pNext) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    auto ptr = MakeShared<ShaderObject>(mContext, shaderName.c_str(),
                                        sourcePath, stage, hasIncludes, defines,
                                        flags, entry, pNext);
    mContext.SetName(ptr->GetHandle(), shaderName.c_str());

    ::std::unique_lock<::std::mutex> lock {mMutex};
    mShaderObjects.emplace(shaderName, ptr);
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

void ShaderManager::ReleaseShaderObject(const char* name,
                                        vk::ShaderStageFlagBits stage,
                                        Type_ShaderMacros const& defines,
                                        const char* entry) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    ::std::unique_lock<::std::mutex> lock {mMutex};
    auto it = mShaderObjects.find(shaderName);
    if (it != mShaderObjects.end()) {
        ::std::unique_lock<::std::mutex> ll {it->second->GetMutex()};
        mShaderObjects.erase(it);
    }
}

ShaderObject* ShaderManager::GetShaderObject(const char* name,
                                             vk::ShaderStageFlagBits stage,
                                             Type_ShaderMacros const& defines,
                                             const char* entry) {
    auto shaderName = ParseShaderName(name, stage, defines, entry);
    ::std::unique_lock<::std::mutex> lock {mMutex};
    return mShaderObjects.at(shaderName).get();
}

ShaderObject* ShaderManager::GetShaderObject(const char* fullname) {
    ::std::unique_lock<::std::mutex> lock {mMutex};
    return mShaderObjects.at(fullname).get();
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
                                              const char* entry) {
    Type_STLString res {};
    res = name;

    // TODO: hlsl
    res = res + "@glsl460" + vk::to_string(stage).insert(0, "@").c_str();

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

ShaderProgram* ShaderManager::CreateProgram(const char* name,
                                            ShaderBase* comp) {
    auto ptr = MakeShared<ShaderProgram>(comp);
    mPrograms.emplace(name, ptr);
    return ptr.get();
}

ShaderProgram* ShaderManager::CreateProgram(const char* name, ShaderBase* vert,
                                            ShaderBase* frag) {
    auto ptr = MakeShared<ShaderProgram>(vert, frag);
    mPrograms.emplace(name, ptr);
    return ptr.get();
}

ShaderProgram* ShaderManager::CreateProgram(const char* name, ShaderBase* task,
                                            ShaderBase* mesh,
                                            ShaderBase* frag) {
    auto ptr = MakeShared<ShaderProgram>(task, mesh, frag);
    mPrograms.emplace(name, ptr);
    return ptr.get();
}

ShaderProgram* ShaderManager::GetProgram(const char* name) const {
    return mPrograms.at(name).get();
}

ShaderProgram ShaderManager::MakeTempProgram(ShaderObject* comp) {
    return {comp};
}

ShaderProgram ShaderManager::MakeTempProgram(ShaderObject* task,
                                             ShaderObject* mesh,
                                             ShaderObject* frag) {
    return {task, mesh, frag};
}

}  // namespace IntelliDesign_NS::Vulkan::Core