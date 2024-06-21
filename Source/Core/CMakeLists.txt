find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 20)

set (CORE_SRC 
    VulkanHelper.hpp

    Utilities/VulkanUtilities.hpp
    Utilities/VulkanUtilities.cpp

    Engine.hpp
    Engine.cpp
    VulkanImage.hpp
    VulkanImage.cpp
    VulkanDescriptors.hpp
    VulkanDescriptors.cpp
    VulkanPipeline.hpp
    VulkanPipeline.cpp
    VulkanBuffer.hpp
    VulkanBuffer.cpp

    MeshType.hpp
    Mesh.hpp
    Mesh.cpp
)

add_library(Core STATIC
    ${CORE_SRC}
)

set_target_properties(Core PROPERTIES FOLDER ${LIB_FOLDER_NAME})

if(MSVC AND NOT USE_CLANG)
    target_compile_options(Core 
        PUBLIC 
            "/MP")
endif()

target_compile_definitions(Core 
    PUBLIC 
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS)
        
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_definitions(Core 
    PUBLIC 
        DEBUG)
else ()
target_compile_definitions(Core 
    PUBLIC 
        NDEBUG)
endif()

# <TODO>
target_compile_definitions(Core 
    PUBLIC 
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX)

target_compile_definitions(Core 
    PUBLIC 
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 )

target_link_libraries(Core 
    PUBLIC
        Vulkan::Headers)

target_include_directories(Core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/Source)

target_include_directories(Core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/External/Output/Include/SDL/include)
        
add_library(SDL_DEBUG SHARED IMPORTED)
add_library(SDL_RELEASE SHARED IMPORTED)

if (MSVC)
    set_target_properties(SDL_DEBUG PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2d.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Debug/SDL2d.lib)
    set_target_properties(SDL_RELEASE PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Release/SDL2.lib)
endif(MSVC)

target_link_libraries(Core
    PUBLIC
        debug SDL_DEBUG
        optimized SDL_RELEASE)


set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")

target_include_directories(Core
    PUBLIC
        ${INTELLIGENDESIGN_CODE_PATH}
)

add_library(INTELLI_DS_MemoryPool_DEBUG STATIC IMPORTED)
add_library(INTELLI_DS_MemoryPool_RELEASE STATIC IMPORTED)

set (INTELLIGENDESIGN_DEMO_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsDemo/Repository")

if (MSVC)
    set_target_properties(INTELLI_DS_MemoryPool_DEBUG 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Debug/LibMemoryPool.lib
            IMPORTED_IMPLIB ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Debug/LibMemoryPool.lib)
    set_target_properties(INTELLI_DS_MemoryPool_RELEASE 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Release/LibMemoryPool.lib
            IMPORTED_IMPLIB $${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Release/LibMemoryPool.lib)
endif(MSVC)

target_link_libraries(Core
    PUBLIC
        debug INTELLI_DS_MemoryPool_DEBUG
        optimized INTELLI_DS_MemoryPool_RELEASE)

target_precompile_headers(Core 
    PUBLIC 
        <functional>
        <deque>
        <optional> 
        <vector> 
        <memory> 
        <string> 
        <thread>
        <span>
        <fstream>
        <unordered_map>)