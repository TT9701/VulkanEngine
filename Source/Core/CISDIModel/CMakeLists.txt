project(CISDI_ModelData)

set(CMAKE_CXX_STANDARD 20)

set (MODEL_SRC
    Common.h
    CISDI_3DModelData.h
    CISDI_3DModelData.cpp
    FBX_Importer.h
    FBX_Importer.cpp
    Assimp_Importer.h
    Assimp_Importer.cpp
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
