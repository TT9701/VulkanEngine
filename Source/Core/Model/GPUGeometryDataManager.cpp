#include "GPUGeometryDataManager.h"

#include "Core/Utilities/Logger.h"
#include "GPUGeometryData.h"

namespace IntelliDesign_NS::Vulkan::Core {

GPUGeometryDataManager::GPUGeometryDataManager(
    VulkanContext& context, ::std::pmr::memory_resource* pMemPool)
    : mContext(context), pMemPool(pMemPool), mGeometries(pMemPool) {}

GPUGeometryData& GPUGeometryDataManager::CreateGPUGeometryData(
    ModelData::CISDI_3DModel const& model) {
    auto ptr = MakeShared<GPUGeometryData>(mContext, model);
    mGeometries.emplace(ptr->GetName(), ptr);
    return *ptr;
}

GPUGeometryData& GPUGeometryDataManager::GetGPUGeometryData(
    const char* name) const {
    return *mGeometries.at(name);
}

void GPUGeometryDataManager::RemoveGPUGeometryData(const char* name) {
    if (mGeometries.contains(name)) {
        mGeometries.erase(name);
    } else {
        DBG_LOG_INFO(
            "GPUGeometryDataManager::RemoveGPUGeometryData: GPUGeometryData %s "
            "not found.",
            name);
    }
}

}  // namespace IntelliDesign_NS::Vulkan::Core