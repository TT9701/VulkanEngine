add_library(imgui STATIC)

target_include_directories(imgui PUBLIC .)

target_sources(imgui PRIVATE
    imconfig.h
    
    imgui.h
    imgui.cpp
    
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_internal.h
    imgui_tables.cpp
    imgui_widgets.cpp
    
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
    
    backends/imgui_impl_sdl2.h
    backends/imgui_impl_sdl2.cpp
    backends/imgui_impl_vulkan.h
    backends/imgui_impl_vulkan.cpp)

    target_link_libraries(imgui PRIVATE Vulkan::Vulkan SDL2::SDL2)