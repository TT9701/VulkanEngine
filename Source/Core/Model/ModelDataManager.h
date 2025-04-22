#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::ModelData {

struct CISDI_3DModel;

class ModelDataManager {
    // key = model path in disk, value = shared ptr to model
    using Type_ModelMap = Core::MemoryPool::Type_STLUnorderedMap_String<
        Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>>;

    // key = model name, value = model path in disk
    using Type_ModelPathMap = Core::MemoryPool::Type_STLUnorderedMap_String<
        Core::MemoryPool::Type_STLString>;

public:
    explicit ModelDataManager(::std::pmr::memory_resource* pMemPool);

    Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const> Create_CISDI_3DModel(
        const char* path);

    Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>
    Get_CISDI_3DModel_FromName(const char* name) const;

    Core::MemoryPool::Type_SharedPtr<CISDI_3DModel const>
    Get_CISDI_3DModel_FromPath(const char* path) const;

    // return empty string if model does not exist.
    Core::MemoryPool::Type_STLString Get_CISDI_3DModel_Path(const char* name);

    Core::MemoryPool::Type_STLString Get_CISDI_3DModel_Path(
        CISDI_3DModel const& model);

    void Remove_CISDI_3DModel(const char* name);

private:
    ::std::pmr::memory_resource* pMemPool;

    Type_ModelMap mModels;
    Type_ModelPathMap mPaths;
};

}  // namespace IntelliDesign_NS::ModelData
