#include "ModelDataManager.h"

#include <filesystem>

#include "CISDI_3DModelData.h"
#include "Core/Utilities/Logger.h"

namespace IntelliDesign_NS::ModelData {

ModelDataManager::ModelDataManager(std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool) {}

Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>
ModelDataManager::Create_CISDI_3DModel(const char* path) {
    ZoneScopedS(10);

    auto modelPath = ::std::filesystem::path {path};

    if (mModels.contains(path))
        return mModels.at(path);

    auto ptr = Core::MemoryPool::New_Shared<CISDI_3DModel>(pMemPool, pMemPool);

    auto emplacePtr =
        [&](Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>&& ptr) {
            mPaths.emplace(ptr->name, path);

            auto [it, success] = mModels.emplace(path, ::std::move(ptr));
            if (success)
                return it->second;
            throw ::std::runtime_error("");
        };

    if (modelPath.extension() == CISDI_3DModel_Subfix_Str) {
        {
            ZoneScopedNS("Load CISDI Model", 10);
            Load(ptr.get(), path, pMemPool);
        }
        return emplacePtr(::std::move(ptr));
    }

    {
        ZoneScopedNS("Convert CISDI Model", 10);
        Convert(ptr.get(), path, false, pMemPool, nullptr);
    }
    return emplacePtr(::std::move(ptr));
}

Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>
ModelDataManager::Get_CISDI_3DModel_FromName(const char* name) const {
    if (mPaths.contains(name)) {
        auto const& path = mPaths.at(name);
        if (mModels.contains(path))
            return mModels.at(path);
    }

    return nullptr;
}

Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>
ModelDataManager::Get_CISDI_3DModel_FromPath(const char* path) const {
    if (mModels.contains(path))
        return mModels.at(path);
    return nullptr;
}

Core::MemoryPool::Type_STLString ModelDataManager::Get_CISDI_3DModel_Path(
    const char* name) {
    if (mPaths.contains(name))
        return mPaths.at(name);
    return {};
}

Core::MemoryPool::Type_STLString ModelDataManager::Get_CISDI_3DModel_Path(
    CISDI_3DModel const& model) {
    for (auto const& [path, m] : mModels) {
        if (m.get() == &model) {
            return path;
        }
    }
    return {};
}

void ModelDataManager::Remove_CISDI_3DModel(const char* name) {
    if (mPaths.contains(name)) {
        auto const& path = mPaths.at(name);
        if (mModels.contains(path)) {
            ZoneScopedNS("Remove_CISDI_3DModel", 10);
            mModels.erase(path);
        }
        mPaths.erase(name);
    } else {
        DBG_LOG_INFO(
            "ModelDataManager::Remove_CISDI_3DModel: Model %s not found.",
            name);
    }
}

}  // namespace IntelliDesign_NS::ModelData