cmake_minimum_required (VERSION 3.20)

set_property (GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT CMAKE_BUILD_TYPE)
    message ("Setting build type to 'Debug' as none was specified.")
    set (CMAKE_BUILD_TYPE Debug)
endif ()

project (VulkanEngine)

set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/Install")

if (CUDA_VULKAN_INTEROP)
    enable_language (CUDA)
endif ()

option (TRACY_ENABLE "" ON)

find_package (Vulkan COMPONENTS shaderc_combined SPIRV-Tools REQUIRED)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin")

if (${CMAKE_GENERATOR} MATCHES "Ninja")
    if (Win32)
        set (CMAKE_C_COMPILER "cl")
        set (CMAKE_CXX_COMPILER "cl")
    endif ()

    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}")
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}")
endif ()

if (${CMAKE_CXX_COMPILER} MATCHES "cl")
    add_compile_options ("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options ("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    cmake_host_system_information (RESULT CPU_NUMBER_OF_LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    add_compile_options ($<$<C_COMPILER_ID:MSVC>:/MP${CPU_NUMBER_OF_LOGICAL_CORES}>)
    add_compile_options ($<$<CXX_COMPILER_ID:MSVC>:/MP${CPU_NUMBER_OF_LOGICAL_CORES}>)
endif ()

set (LIB_FOLDER_NAME "Libs")
set (DLL_FOLDER_NAME "Dlls")
set (EXECUTABLE_FOLDER_NAME "Demos")
set (CUSTOM_FOLDER_NAME "Custom_Projects")

add_subdirectory (External)
add_subdirectory (Source)

find_program (GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file (GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/Shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/Shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/Shaders/*.comp"
    "${PROJECT_SOURCE_DIR}/Shaders/*.task"
    "${PROJECT_SOURCE_DIR}/Shaders/*.mesh"
)

foreach (GLSL ${GLSL_SOURCE_FILES})
    message (STATUS "BUILDING SHADER")
    get_filename_component (FILE_NAME ${GLSL} NAME)
    set (SPIRV "${PROJECT_SOURCE_DIR}/Shaders/${FILE_NAME}.spv")
    message (STATUS ${GLSL})
    add_custom_command (
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV} -g --target-env spirv1.4 --target-env vulkan1.2
        DEPENDS ${GLSL})
    list (APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach (GLSL)

add_custom_target (
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
)

set_target_properties (Shaders PROPERTIES FOLDER ${CUSTOM_FOLDER_NAME})

add_subdirectory (Demos)