add_executable (Runtime 
    main.cpp
)

set_property(TARGET Runtime 
    PROPERTY 
        CXX_STANDARD 20)
        
set_property(TARGET Runtime 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:Runtime>)