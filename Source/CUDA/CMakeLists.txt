find_package(Vulkan REQUIRED)

set (CUDA_SRC 
    CUDAHelper.h
    
    CUDAStream.h
    CUDAStream.cu
    CUDASurface.h
    CUDASurface.cu

    CUDAVulkan.h
    CUDAVulkan.cu)

add_library(CUDA STATIC ${CUDA_SRC})

set_property(TARGET CUDA 
    PROPERTY 
        CXX_STANDARD 20)

set_property(TARGET CUDA 
    PROPERTY 
        CUDA_STANDARD 20)

target_link_libraries(CUDA 
    PRIVATE
        Vulkan::Vulkan)

target_link_libraries(CUDA
    PRIVATE
        GPUOpen::VulkanMemoryAllocator)

set_target_properties(CUDA PROPERTIES FOLDER ${LIB_FOLDER_NAME})

target_compile_definitions(CUDA 
    PUBLIC 
        CUDA_VULKAN_INTEROP) 