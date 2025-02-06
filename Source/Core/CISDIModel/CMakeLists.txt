project(CISDI_ModelData)

set(CMAKE_CXX_STANDARD 20)

set (MODEL_SRC
    CISDI_3DModelData.h
    Source/Common/Common.h
    Source/Common/Math.h
    Source/Common/BaseTypes.h
    Source/Common/FileIO.h
    Source/Common/FileIO.cpp
    Source/Common/PMR_Def.h
    Source/Common/CodeGenDef.h
    Source/CISDI_3DModelData.cpp
    Source/Importer/FBX_Importer.h
    Source/Importer/FBX_Importer.cpp
    Source/Importer/Assimp_Importer.h
    Source/Importer/Assimp_Importer.cpp
    Source/Importer/Combined_Importer.h
    Source/Importer/Combined_Importer.cpp
)

add_library(CISDI_ModelData SHARED ${MODEL_SRC})

set_target_properties(CISDI_ModelData 
    PROPERTIES
        CXX_STANDARD 20)

target_include_directories(CISDI_ModelData PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(CISDI_ModelData 
    PRIVATE 
        CISDI_MODEL_DATA_EXPORTS)

target_link_libraries(CISDI_ModelData
    PUBLIC
    assimp::assimp
    meshoptimizer
    fbx::sdk)
