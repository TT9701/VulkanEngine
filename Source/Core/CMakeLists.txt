find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_library(Core STATIC
    VulkanHelper.hpp

    Engine.hpp
    Engine.cpp
)

if(MSVC AND NOT USE_CLANG)
    target_compile_options(Core 
        PUBLIC 
            "/MP")
endif()

target_compile_definitions(Core 
    PUBLIC 
        GLM_FORCE_DEPTH_ZERO_TO_ONE)
        
target_compile_definitions(Core 
    PUBLIC 
        GLM_FORCE_RADIANS)

# TODO
target_compile_definitions(Core 
    PUBLIC 
        VK_USE_PLATFORM_WIN32_KHR)
target_compile_definitions(Core 
    PUBLIC 
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 )

target_link_libraries(Core 
    PUBLIC
        Vulkan::Headers)

target_include_directories(Core 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_precompile_headers(Core 
    PUBLIC 
        <optional> 
        <vector> 
        <memory> 
        <string> 
        <vector> 
        <thread>
        <unordered_map>)