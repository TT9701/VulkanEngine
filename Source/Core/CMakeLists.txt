find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 20)

set (CORE_SRC 
    Utilities/VulkanUtilities.hpp
    Utilities/VulkanUtilities.cpp

    DbgMemLeak.h

    Engine.hpp
    Engine.cpp
)

add_library(Core STATIC
    ${CORE_SRC}
)

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
        
if (MSVC)
    add_library(SDL SHARED IMPORTED)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set_target_properties(SDL PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2d.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Debug/SDL2d.lib)
    else()
        set_target_properties(SDL PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Release/SDL2.lib)
    endif()
endif(MSVC)

target_link_libraries(Core
    PUBLIC
        SDL)

target_precompile_headers(Core 
    PUBLIC 
        <optional> 
        <vector> 
        <memory> 
        <string> 
        <vector> 
        <functional>
        <thread>
        <unordered_map>)