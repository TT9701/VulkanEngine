#include "ModelDataManager.h"

#include <filesystem>

#include "CISDI_3DModelData.h"
#include "Core/Utilities/Logger.h"

namespace IntelliDesign_NS::ModelData {

ModelDataManager::ModelDataManager(std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool) {}

Core::MemoryPool::Type_SharedPtr<CISDI_3DModel>
ModelDataManager::Create_CISDI_3DModel(const char* path) {
    auto modelPath = ::std::filesystem::path {path};

    ::std::u8string modelName {};
    if (modelPath.extension() == ".cisdi")
        modelName = modelPath.stem().stem().u8string();
    else
        modelName = modelPath.stem().u8string();

    if (mModels.contains(modelName.c_str()))
        return mModels.at(modelName.c_str());

    auto ptr = Core::MemoryPool::New_Shared<CISDI_3DModel>(pMemPool, pMemPool);

    Type_STLString name {};

    auto insertData =
        [&](Core::MemoryPool::Type_SharedPtr<CISDI_3DModel>&& model,
            Type_STLString& n) {
            n = model->name;
            if (mModels.contains(model->name)) {
                mModels.erase(model->name);
            }
            mModels.emplace(model->name, ::std::move(model));
        };

    if (modelPath.extension() == CISDI_3DModel_Subfix_Str) {
        Load(ptr.get(), modelPath.string().c_str(), pMemPool);
        insertData(::std::move(ptr), name);
        return mModels.at(name);
    } else {
        Convert(ptr.get(), modelPath.string().c_str(), false, pMemPool,
                nullptr);
        insertData(::std::move(ptr), name);
        return mModels.at(name);
    }
}

Core::MemoryPool::Type_SharedPtr<CISDI_3DModel>
ModelDataManager::Get_CISDI_3DModel(const char* name) const {
    if (mModels.contains(name))
        return mModels.at(name);
    else
        return nullptr;
}

void ModelDataManager::Remove_CISDI_3DModel(const char* name) {
    if (mModels.contains(name))
        mModels.erase(name);
    else {
        DBG_LOG_INFO(
            "ModelDataManager::Remove_CISDI_3DModel: Model %s not found.",
            name);
    }
}

}  // namespace IntelliDesign_NS::ModelData