#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::ModelData {

struct CISDI_3DModel;

class ModelDataManager {
public:
    explicit ModelDataManager(::std::pmr::memory_resource* pMemPool);

    Core::MemoryPool::Type_SharedPtr<CISDI_3DModel> Create_CISDI_3DModel(
        const char* path);

    Core::MemoryPool::Type_SharedPtr<CISDI_3DModel> Get_CISDI_3DModel(
        const char* name) const;

    void Remove_CISDI_3DModel(const char* name);

private:
    ::std::pmr::memory_resource* pMemPool;

    Core::MemoryPool::Type_STLUnorderedMap_String<
        Core::MemoryPool::Type_SharedPtr<CISDI_3DModel>>
        mModels;
};

}  // namespace IntelliDesign_NS::ModelData
