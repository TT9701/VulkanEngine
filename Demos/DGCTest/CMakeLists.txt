add_executable (DGCTest 
    DGCTest.cpp
    DGCTest.h
)

set_property(TARGET DGCTest 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET DGCTest 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:DGCTest>)

set_target_properties(DGCTest PROPERTIES FOLDER ${EXECUTABLE_FOLDER_NAME})

target_link_libraries(DGCTest
    PUBLIC
        Core)