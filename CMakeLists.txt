cmake_minimum_required(VERSION 3.18)
project(CUDAOP LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)

# Target SM 120 for 5070 Ti
set(CMAKE_CUDA_ARCHITECTURES 120)

# Linux / WSL2 Configuration
if (UNIX)
    set(CUDA_ROOT "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
    
    include_directories("${CUDA_ROOT}/include")
    include_directories("${PROJECT_SOURCE_DIR}")
    include_directories("${PROJECT_SOURCE_DIR}/op")
    link_directories("${CUDA_ROOT}/lib64")
    
    if(EXISTS "/usr/lib/wsl/lib")
        link_directories("/usr/lib/wsl/lib")
    endif()
endif()

# Find CUDA and related libraries
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY cudnn PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 REQUIRED)
find_path(CUDNN_INCLUDE cudnn.h PATHS /usr/include /usr/local/cuda/include REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
)

# --- Build Explicitly Specified .cu Files ---
set(OP_SOURCES
    "op/conv/conv2d_k128.cu"
    "op/conv/conv2d_k64_u8.cu"
    "op/conv/conv2d_k64_u4.cu"
    "op/conv/conv2d_sn_k64_u8.cu"
    "op/conv/conv2d_k64_u8_fp16.cu"
    "op/conv/conv2d_sn_k64_u8_fp16.cu"
    "op/conv/conv2d_erf_k64_u8.cu"
    "op/conv/conv2d_erf_k64_u8_fp16.cu"
    "op/resize/resize_u8.cu"
    "op/mul/mul_u8.cu"
    "op/linear/linear_k128.cu"
    "op/linear/linear_sn_k64_u8.cu"
    "op/linear/linear_sn_k64_u8_fp16.cu"
)

foreach(SRC_FILE ${OP_SOURCES})
    # 获取不带后缀的文件名作为可执行文件名
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
    
    # 将该文件编译为独立的可执行程序
    add_executable(${EXE_NAME} ${SRC_FILE})
    
    # 链接基础 CUDA 库
    target_link_libraries(${EXE_NAME} PRIVATE 
        CUDA::cudart 
        CUDA::cuda_driver 
        CUDA::cublas
    )

    if(TARGET CUDA::cudnn)
        target_link_libraries(${EXE_NAME} PRIVATE CUDA::cudnn)
    elseif(CUDNN_LIBRARY)
        target_link_libraries(${EXE_NAME} PRIVATE ${CUDNN_LIBRARY})
        target_include_directories(${EXE_NAME} PRIVATE ${CUDNN_INCLUDE})
    endif()
    
    # 编译选项
    target_compile_options(${EXE_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-Wno-deprecated-gpu-targets>
    )
    
    message(STATUS "Added standalone target: ${EXE_NAME}")
endforeach()
