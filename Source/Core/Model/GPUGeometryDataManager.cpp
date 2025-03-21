#include "GPUGeometryDataManager.h"

#include "Core/Utilities/Logger.h"
#include "GPUGeometryData.h"

namespace IntelliDesign_NS::Vulkan::Core {

GPUGeometryDataManager::GPUGeometryDataManager(
    VulkanContext& context, uint32_t dgcSequenceMaxDrawCount,
    ::std::pmr::memory_resource* pMemPool)
    : mContext(context),
      mDGCSequenceMaxDrawCount(dgcSequenceMaxDrawCount),
      pMemPool(pMemPool),
      mGeometries(pMemPool) {}

SharedPtr<GPUGeometryData> GPUGeometryDataManager::CreateGPUGeometryData(
    ModelData::CISDI_3DModel const& model) {
    if (mGeometries.contains(model.name)) 
        return mGeometries.at(model.name);
    
    auto ptr =
        MakeShared<GPUGeometryData>(mContext, model, mDGCSequenceMaxDrawCount);
    mGeometries.emplace(ptr->GetName(), ptr);

    return ptr;
}

SharedPtr<GPUGeometryData> GPUGeometryDataManager::GetGPUGeometryData(
    const char* name) const {
    return mGeometries.at(name);
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