add_executable (CISDIModel_Generator 
    CISDIModel_Generator.cpp
)

set_property(TARGET CISDIModel_Generator 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET CISDIModel_Generator 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CISDIModel_Generator>)

set_target_properties(CISDIModel_Generator PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})

target_link_libraries(CISDIModel_Generator
    PUBLIC
        CISDI_ModelData)

add_executable(CISDIModel_BatchGenerator
    CISDIModel_BatchGenerator.cpp
    CISDIModel_BatchGenerator.h
)

set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")
set (INTELLIGENDESIGN_LIB_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/LibraryOutputs/x64")

target_include_directories (CISDIModel_BatchGenerator
    PUBLIC
    ${INTELLIGENDESIGN_CODE_PATH}
)

set_property(TARGET CISDIModel_BatchGenerator 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET CISDIModel_BatchGenerator 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CISDIModel_BatchGenerator>)

set_target_properties(CISDIModel_BatchGenerator PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})


if (NOT TARGET INTELLI_DS_MemoryPool)
    add_library (INTELLI_DS_MemoryPool STATIC IMPORTED)
endif ()

set_target_properties (INTELLI_DS_MemoryPool
    PROPERTIES
    IMPORTED_LOCATION_DEBUG ${INTELLIGENDESIGN_LIB_PATH}/Debug/LibMemoryPool.lib
    IMPORTED_LOCATION ${INTELLIGENDESIGN_LIB_PATH}/Release/LibMemoryPool.lib)

target_link_libraries (CISDIModel_BatchGenerator
        PUBLIC
        INTELLI_DS_MemoryPool)
