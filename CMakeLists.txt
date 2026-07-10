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
        ${PROJECT_SOURCE_DIR}/op
)

# --- Build Explicitly Specified .cu Files ---
set(OP_SOURCES
    "op/device_query.cu"
    "op/conv/conv2d_k128.cu"
    "op/conv/conv2d_k64_u8.cu"
    "op/conv/conv2d_k64_u4.cu"
    "op/conv/conv2d_sn_k64_u8.cu"
    "op/conv/conv2d_k64_u8_fp16.cu"
    "op/conv/conv2d_sn_k64_u8_fp16.cu"
    "op/conv/conv2d_erf_k64_u8.cu"
    "op/conv/conv2d_erf_k64_u8_fp16.cu"
    "op/dwconv/conv2d_groups_fp32.cu"
    "op/dwconv/dwconv_k128.cu"
    "op/dwconv/dwconv_k128_fp16.cu"
    "op/dwconv/dwconv_k32.cu"
    "op/dwconv/dwconv_k32_fp16.cu"
    "op/dwconv/dwconv_k64.cu"
    "op/dwconv/dwconv_k64_fp16.cu"
    "op/dwconv/dwconv_spike_k32.cu"
    "op/dwconv/dwconv_spike_k32_fp16.cu"
    "op/dwconv/dwconv_spike_k64.cu"
    "op/dwconv/dwconv_spike_k64_fp16.cu"
    "op/dwconv/dwconv_spike_k128.cu"
    "op/dwconv/dwconv_spike_k128_fp16.cu"
    "op/dwconv/opt_conv2d_groups_fp32.cu"
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

    if(EXE_NAME STREQUAL "opt_conv2d_groups_fp32")
        target_sources(${EXE_NAME} PRIVATE op/cpu/cpu_ops.cpp)
    endif()
    
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

# Build the PyTorch CUDA extension from Python. This target is intentionally
# opt-in because it depends on the active Python environment and PyTorch.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(lora_moe_ops
    COMMAND
        ${CMAKE_COMMAND} -E env
        TORCH_CUDA_ARCH_LIST=12.0
        ${Python3_EXECUTABLE} setup.py build_ext --inplace
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/lora_moe"
    COMMENT
        "Building the lora_moe_ops Python CUDA extension"
    VERBATIM
)

add_custom_target(lora_moe_standard_test
    COMMAND
        ${Python3_EXECUTABLE} test_standard.py
    COMMAND
        ${Python3_EXECUTABLE} test_nonstandard.py
    COMMAND
        ${Python3_EXECUTABLE} test_grouped_gemm.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/lora_moe"
    DEPENDS
        lora_moe_ops
    COMMENT
        "Testing the LoRA-MoE Torch implementations"
    VERBATIM
)

set(
    CUTLASS_ROOT
    "${PROJECT_SOURCE_DIR}/../cutlass"
    CACHE PATH
    "Path to the CUTLASS source tree"
)

add_executable(
    lora_moe_cutlass_grouped_test
    op/lora_moe/csrc/cutlass_grouped_gemm_test.cu
)
target_include_directories(
    lora_moe_cutlass_grouped_test
    PRIVATE
        "${CUTLASS_ROOT}/include"
)
target_link_libraries(
    lora_moe_cutlass_grouped_test
    PRIVATE
        CUDA::cudart
)
