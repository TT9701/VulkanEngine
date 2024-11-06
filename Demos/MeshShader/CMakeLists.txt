add_executable (MeshShaderDemo 
    MeshShaderDemo.cpp
    MeshShaderDemo.h
)

set_property(TARGET MeshShaderDemo 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET MeshShaderDemo 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:MeshShaderDemo>)

set_target_properties(MeshShaderDemo PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})

target_link_libraries(MeshShaderDemo
    PUBLIC
        Core)