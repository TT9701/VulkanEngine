find_package(Vulkan REQUIRED)

set (CUDA_SRC 
    CUDAVulkan.h
    CUDAVulkan.cu)

add_library(CUDA ${CUDA_SRC})

set_property(TARGET CUDA 
    PROPERTY 
        CXX_STANDARD 20)

target_link_libraries(CUDA 
    PRIVATE
        Vulkan::Vulkan)