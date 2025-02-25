#pragma once

#include "Core/Utilities/MemoryPool.h"

namespace IntelliDesign_NS::ModelData {
struct CISDI_3DModel;
}

namespace IntelliDesign_NS::Vulkan::Core {

class VulkanContext;
class GPUGeometryData;

class GPUGeometryDataManager {
public:
    GPUGeometryDataManager(VulkanContext& context,
                           ::std::pmr::memory_resource* pMemPool);

    GPUGeometryData& CreateGPUGeometryData(
        ModelData::CISDI_3DModel const& model);

    GPUGeometryData& GetGPUGeometryData(const char* name) const;

    void RemoveGPUGeometryData(const char* name);

private:
    VulkanContext& mContext;
    ::std::pmr::memory_resource* pMemPool;

    Type_STLUnorderedMap_String<SharedPtr<GPUGeometryData>> mGeometries;
};

}  // namespace IntelliDesign_NS::Vulkan::Core