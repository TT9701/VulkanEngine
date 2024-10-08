
if (CUDA_VULKAN_INTEROP)
    add_subdirectory(CUDA)
endif()

add_subdirectory(Core)

if (CUDA_VULKAN_INTEROP)
target_link_libraries(Core
    PUBLIC
        CUDA)
endif()