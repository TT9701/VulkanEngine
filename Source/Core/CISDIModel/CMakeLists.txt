project (CISDI_ModelData)

set (CMAKE_CXX_STANDARD 20)

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

add_library (CISDI_ModelData SHARED ${MODEL_SRC})

set_target_properties (CISDI_ModelData
    PROPERTIES
    CXX_STANDARD 20)

target_include_directories (CISDI_ModelData PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories (CISDI_ModelData
    PUBLIC
    ${INTELLIGENDESIGN_CODE_PATH}
)

target_compile_definitions (CISDI_ModelData
    PRIVATE
    CISDI_MODEL_DATA_EXPORTS)

set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")
set (INTELLIGENDESIGN_LIB_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/LibraryOutputs/x64")

set (INTELLIGENDESIGN_EXTERNAL_ASSIMP_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/AssimpTools/Include")
set (INTELLIGENDESIGN_EXTERNAL_MESH_OPTIMIZER_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/meshoptimizer")
set (INTELLIGENDESIGN_EXTERNAL_FBX_SDK_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/FBXSDK")

set (INTELLIGENDESIGN_EXTERNAL_LIB_PATH "${PROJECT_SOURCE_DIR}/../../../../IntelligenDesign/Git-IntelligenDesignToolsBinaries/Repository/3rdParty/x64")

target_include_directories (CISDI_ModelData
    PUBLIC
    ${INTELLIGENDESIGN_EXTERNAL_FBX_SDK_INCLUDE_PATH}
    ${INTELLIGENDESIGN_EXTERNAL_ASSIMP_INCLUDE_PATH}
    ${INTELLIGENDESIGN_EXTERNAL_MESH_OPTIMIZER_INCLUDE_PATH}
    ${INTELLIGENDESIGN_CODE_PATH}
)

if (NOT TARGET INTELLI_DS_MemoryPool)
    add_library (INTELLI_DS_MemoryPool STATIC IMPORTED)
endif ()

set_target_properties (INTELLI_DS_MemoryPool
    PROPERTIES
    IMPORTED_LOCATION_DEBUG ${INTELLIGENDESIGN_LIB_PATH}/Debug/LibMemoryPool.lib
    IMPORTED_LOCATION ${INTELLIGENDESIGN_LIB_PATH}/Release/LibMemoryPool.lib)

function (AddExternalLibrary targetName type libName)
    if (NOT TARGET ${targetName})
        add_library (${targetName} ${type} IMPORTED)
        if (type STREQUAL "SHARED")
            set_target_properties (${targetName} PROPERTIES 
                IMPORTED_LOCATION_DEBUG ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}d.dll
                IMPORTED_IMPLIB_DEBUG ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}d.lib
                IMPORTED_LOCATION ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}.dll
                IMPORTED_IMPLIB ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}.lib)
        else ()
            set_target_properties (${targetName} PROPERTIES 
                IMPORTED_LOCATION_DEBUG ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}d.lib
                IMPORTED_LOCATION ${INTELLIGENDESIGN_EXTERNAL_LIB_PATH}/${libName}.lib)
        endif()
    endif ()
endfunction()

AddExternalLibrary(EXTERNAL_ASSIMP SHARED "assimp-vc143-mt")
AddExternalLibrary(EXTERNAL_MESHOPTIMIZER SHARED "meshoptimizer")
AddExternalLibrary(EXTERNAL_FBXSDK SHARED "libfbxsdk")

target_compile_definitions(EXTERNAL_FBXSDK INTERFACE FBXSDK_SHARED)


target_link_libraries (CISDI_ModelData
    PRIVATE
    INTELLI_DS_MemoryPool
    EXTERNAL_ASSIMP
    EXTERNAL_MESHOPTIMIZER
    EXTERNAL_FBXSDK)
