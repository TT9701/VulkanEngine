add_subdirectory(Core)

add_subdirectory(Runtime)

target_link_libraries(Runtime
    PUBLIC
        Core)