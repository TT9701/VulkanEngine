#pragma once

#include <filesystem>

namespace IntelliDesign_NS::ModelData {

struct CISDI_3DModel;

::std::string ProcessOutputPath(const char* input, const char* output);

void Write_CISDI_File(const char* outputPath, CISDI_3DModel const& data);

CISDI_3DModel Read_CISDI_File(const char* path,
                              ::std::pmr::memory_resource* pMemPool);

}  // namespace IntelliDesign_NS::ModelData