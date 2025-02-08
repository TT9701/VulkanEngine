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

set_property(TARGET CISDIModel_BatchGenerator 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET CISDIModel_BatchGenerator 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CISDIModel_BatchGenerator>)

set_target_properties(CISDIModel_BatchGenerator PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})
