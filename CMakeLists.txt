cmake_minimum_required (VERSION 3.20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
    message("Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug)
endif()

project (VulkanEngine)

enable_language(CUDA)

find_package (Vulkan COMPONENTS glslang shaderc_combined SPIRV-Tools REQUIRED)

if(${CMAKE_GENERATOR} MATCHES "Visual Studio*")
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin")
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin")
else()
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}")
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}")
endif()

set (LIB_FOLDER_NAME "Libs")
set (DLL_FOLDER_NAME "Dlls")
set (EXECUTABLE_FOLDER_NAME "Executables")
set (CUSTOM_FOLDER_NAME "Custom_Projects")

add_subdirectory (External)
add_subdirectory (Source)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/Shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/Shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/Shaders/*.comp"
    "${PROJECT_SOURCE_DIR}/Shaders/*.task"
    "${PROJECT_SOURCE_DIR}/Shaders/*.mesh"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
    message(STATUS "BUILDING SHADER")
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${PROJECT_SOURCE_DIR}/Shaders/${FILE_NAME}.spv")
    message(STATUS ${GLSL})
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV} -g --target-env spirv1.4 --target-env vulkan1.2
        DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
)

set_target_properties(Shaders PROPERTIES FOLDER ${CUSTOM_FOLDER_NAME})

set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
    PROPERTY 
        VS_STARTUP_PROJECT Runtime)