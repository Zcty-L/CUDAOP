cmake_minimum_required(VERSION 3.18)

# Query the current GPU on every configure so that a stale CMake cache from
# another platform cannot pin the old SM. Callers that intentionally
# cross-compile can use -DCUDAOP_CUDA_ARCHITECTURES=<arch>.
set(
    CUDAOP_CUDA_ARCHITECTURES
    ""
    CACHE STRING
    "Explicit CUDA architectures; empty selects the current GPU"
)
if(CUDAOP_CUDA_ARCHITECTURES)
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${CUDAOP_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures"
        FORCE
    )
else()
    execute_process(
        COMMAND
            nvidia-smi
            --query-gpu=compute_cap
            --format=csv,noheader,nounits
        RESULT_VARIABLE CUDAOP_GPU_QUERY_RESULT
        OUTPUT_VARIABLE CUDAOP_GPU_COMPUTE_CAPS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT CUDAOP_GPU_QUERY_RESULT EQUAL 0 OR NOT CUDAOP_GPU_COMPUTE_CAPS)
        message(
            FATAL_ERROR
            "Cannot query the current GPU compute capability. "
            "Use -DCUDAOP_CUDA_ARCHITECTURES=<arch> explicitly."
        )
    endif()

    string(REPLACE "\r" "" CUDAOP_GPU_COMPUTE_CAPS "${CUDAOP_GPU_COMPUTE_CAPS}")
    string(REPLACE "\n" ";" CUDAOP_GPU_COMPUTE_CAPS "${CUDAOP_GPU_COMPUTE_CAPS}")
    list(FILTER CUDAOP_GPU_COMPUTE_CAPS EXCLUDE REGEX "^$")
    list(REMOVE_DUPLICATES CUDAOP_GPU_COMPUTE_CAPS)
    list(LENGTH CUDAOP_GPU_COMPUTE_CAPS CUDAOP_GPU_ARCH_COUNT)
    if(NOT CUDAOP_GPU_ARCH_COUNT EQUAL 1)
        message(
            FATAL_ERROR
            "Detected GPUs with different compute capabilities: "
            "${CUDAOP_GPU_COMPUTE_CAPS}. Use "
            "-DCUDAOP_CUDA_ARCHITECTURES=<arch> for the benchmark GPU."
        )
    endif()

    list(GET CUDAOP_GPU_COMPUTE_CAPS 0 CUDAOP_GPU_COMPUTE_CAP)
    string(REPLACE "." "" CUDAOP_NATIVE_CUDA_ARCH "${CUDAOP_GPU_COMPUTE_CAP}")
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${CUDAOP_NATIVE_CUDA_ARCH}"
        CACHE STRING
        "CUDA architectures"
        FORCE
    )
endif()

project(CUDAOP LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)

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
    "op/dwconv/opt_conv2d_groups_fp16.cu"
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

# Compare separate Conv2D + LIF kernels against the fused Conv2D-LIF kernel.
add_executable(
    conv2d_lif_fusion_cmp
    op/conv/conv2d_lif_fusion_cmp.cu
    op/conv/conv2d_k64_u8.cu
    op/conv/conv2d_sn_k64_u8.cu
)
target_compile_definitions(
    conv2d_lif_fusion_cmp
    PRIVATE
        CONV2D_K64_U8_BENCH_INC
        CONV2D_SN_K64_U8_BENCH_INC
)
target_link_libraries(
    conv2d_lif_fusion_cmp
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
)
if(TARGET CUDA::cudnn)
    target_link_libraries(conv2d_lif_fusion_cmp PRIVATE CUDA::cudnn)
elseif(CUDNN_LIBRARY)
    target_link_libraries(conv2d_lif_fusion_cmp PRIVATE ${CUDNN_LIBRARY})
    target_include_directories(
        conv2d_lif_fusion_cmp
        PRIVATE
            ${CUDNN_INCLUDE}
    )
endif()
target_compile_options(
    conv2d_lif_fusion_cmp
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-Wno-deprecated-gpu-targets>
)

# Compare separate FP16 Conv2D + LIF against the fused FP16 kernel.
add_executable(
    conv2d_lif_fusion_cmp_fp16
    op/conv/conv2d_lif_fusion_cmp_fp16.cu
    op/conv/conv2d_k64_u8_fp16.cu
    op/conv/conv2d_sn_k64_u8_fp16.cu
)
target_compile_definitions(
    conv2d_lif_fusion_cmp_fp16
    PRIVATE
        CONV2D_K64_U8_FP16_BENCH_INC
        CONV2D_SN_FP16_BENCH_INC
)
target_link_libraries(
    conv2d_lif_fusion_cmp_fp16
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
)
if(TARGET CUDA::cudnn)
    target_link_libraries(conv2d_lif_fusion_cmp_fp16 PRIVATE CUDA::cudnn)
elseif(CUDNN_LIBRARY)
    target_link_libraries(conv2d_lif_fusion_cmp_fp16 PRIVATE ${CUDNN_LIBRARY})
    target_include_directories(
        conv2d_lif_fusion_cmp_fp16
        PRIVATE
            ${CUDNN_INCLUDE}
    )
endif()
target_compile_options(
    conv2d_lif_fusion_cmp_fp16
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-Wno-deprecated-gpu-targets>
)

# Build the PyTorch CUDA extension from Python. This target is intentionally
# opt-in because it depends on the active Python environment and PyTorch.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(lora_moe_ops
    COMMAND
        ${Python3_EXECUTABLE} build.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/lora_moe"
    COMMENT
        "Building the lora_moe_ops Python CUDA extension"
    VERBATIM
)

# Build the neuron PyTorch CUDA extension on demand. Custom targets are not
# part of the default build unless they are explicitly marked with ALL.
add_custom_target(neuron_ops
    COMMAND
        ${Python3_EXECUTABLE} build.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/neuron"
    COMMENT
        "Building the neuron_ops Python CUDA extension"
    VERBATIM
)

add_custom_target(neuron_ops_test
    COMMAND
        ${Python3_EXECUTABLE} test_ops.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/neuron"
    DEPENDS
        neuron_ops
    COMMENT
        "Testing the neuron_ops Python CUDA extension"
    VERBATIM
)

add_custom_target(neuron_ops_benchmark
    COMMAND
        ${Python3_EXECUTABLE} benchmark_ops.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/neuron"
    DEPENDS
        neuron_ops
    COMMENT
        "Benchmarking neuron_ops against the CuPy implementations"
    VERBATIM
)

add_custom_target(lora_moe_standard_test
    COMMAND
        ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${PROJECT_SOURCE_DIR}/op/grouped_gemm"
        ${Python3_EXECUTABLE} test_standard.py
    COMMAND
        ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${PROJECT_SOURCE_DIR}/op/grouped_gemm"
        ${Python3_EXECUTABLE} test_nonstandard.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/lora_moe"
    DEPENDS
        lora_moe_ops
        cudaop_grouped_gemm
    COMMENT
        "Testing the LoRA-MoE Torch implementations"
    VERBATIM
)

add_custom_target(cudaop_grouped_gemm
    COMMAND
        ${Python3_EXECUTABLE} build.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/grouped_gemm"
    COMMENT
        "Building the cudaop_grouped_gemm Python CUDA extension"
    VERBATIM
)

add_custom_target(cudaop_grouped_gemm_test
    COMMAND
        ${Python3_EXECUTABLE} test_grouped_gemm.py
    WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/op/grouped_gemm"
    DEPENDS
        cudaop_grouped_gemm
    COMMENT
        "Testing cudaop_grouped_gemm"
    VERBATIM
)
