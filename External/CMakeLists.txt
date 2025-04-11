cmake_minimum_required (VERSION 3.20)

project (External)

# set (BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Output/Bin)
# set (LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Output/Lib)
# set (INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Output/Include)

# set (DEBUG_LIB_DIR ${LIB_DIR}/Debug)
# set (RELEASE_LIB_DIR ${LIB_DIR}/Release)
set (SDL_STATIC OFF CACHE BOOL "Disable sdl static library" FORCE)
set (SDL_TEST OFF CACHE BOOL "Disable sdl test" FORCE)
add_subdirectory (SDL EXCLUDE_FROM_ALL)
set_target_properties (SDL2 PROPERTIES FOLDER ${DLL_FOLDER_NAME})
set_target_properties (sdl_headers_copy PROPERTIES FOLDER ${CUSTOM_FOLDER_NAME})

set (ASSIMP_BUILD_TESTS OFF CACHE BOOL "Disable assimp test" FORCE)
add_subdirectory (Assimp EXCLUDE_FROM_ALL)
set_target_properties (assimp PROPERTIES FOLDER ${DLL_FOLDER_NAME})
set_target_properties (zlibstatic PROPERTIES FOLDER ${LIB_FOLDER_NAME})

set (VMA_BUILD_SAMPLES OFF CACHE BOOL "Don't build vma sample" FORCE)
add_subdirectory (vma EXCLUDE_FROM_ALL)

set (MESHOPT_BUILD_SHARED_LIBS ON CACHE BOOL "Build shared lib" FORCE)
add_subdirectory (meshoptimizer EXCLUDE_FROM_ALL)
set_target_properties (meshoptimizer PROPERTIES FOLDER ${DLL_FOLDER_NAME})

set (SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "Disable test" FORCE)
add_subdirectory (SPIRV-Cross EXCLUDE_FROM_ALL)
set_target_properties (spirv-cross-c PROPERTIES FOLDER ${LIB_FOLDER_NAME})

add_subdirectory (imgui)
set_target_properties (imgui PROPERTIES FOLDER ${LIB_FOLDER_NAME})

add_subdirectory (tracy)
set_target_properties (TracyClient PROPERTIES FOLDER ${LIB_FOLDER_NAME})

include (../CMake/FindFBX.cmake)

add_subdirectory (ImGuiFileDialog)

target_link_libraries (ImGuiFileDialog
    PUBLIC
    imgui
)