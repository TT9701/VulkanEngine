add_executable (DynamicLoading 
    DynamicLoading.cpp
    DynamicLoading.h
)

set_property(TARGET DynamicLoading 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET DynamicLoading 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:DynamicLoading>)

set_target_properties(DynamicLoading PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsBinaries/Repository/3rdParty/x64")

set_target_properties(DynamicLoading PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})

target_link_libraries(DynamicLoading
    PUBLIC
        Core)