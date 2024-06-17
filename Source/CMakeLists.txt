add_subdirectory(CUDA)

add_subdirectory(Core)

target_link_libraries(Core
    PUBLIC
        CUDA)

add_subdirectory(Runtime)

target_link_libraries(Runtime
    PUBLIC
        Core)