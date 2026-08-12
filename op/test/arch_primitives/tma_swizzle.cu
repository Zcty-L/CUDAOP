#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace
{

// 三种 swizzle 都使用 8 行，使 tile 大小恰好覆盖各自一个完整的字节级
// pattern 周期：32B * 8 = 256B，64B * 8 = 512B，128B * 8 = 1024B。
constexpr int kRows = 8;

// global tensor 的每行固定为 32 个 uint32_t，即 128B。较窄的 swizzle
// 模式只搬运每行左侧的一部分，但 global-memory row stride 保持 128B。
constexpr int kGlobalColumns = 32;

// TMA swizzle 的基本粒度是 16B。一个 uint4 正好包含 4 个 uint32_t，
// 因此用一个 uint4 表示一个可被 swizzle 重新排列的 16B chunk。
constexpr int kValuesPerChunk = 4;
constexpr int kBlockThreads = 64;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkIterations = 200;
constexpr size_t kInputElements =
    static_cast<size_t>(kRows) * kGlobalColumns;
constexpr size_t kInputBytes = kInputElements * sizeof(uint32_t);

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_driver(CUresult status, const char *operation)
{
    if (status == CUDA_SUCCESS)
    {
        return;
    }

    const char *error_name = nullptr;
    const char *error_message = nullptr;
    cuGetErrorName(status, &error_name);
    cuGetErrorString(status, &error_message);

    throw std::runtime_error(
        std::string(operation) + ": " +
        (error_name == nullptr ? "unknown" : error_name) + " (" +
        (error_message == nullptr ? "no description" : error_message) +
        ")");
}

uint32_t make_input_value(int row, int column)
{
    return static_cast<uint32_t>(row * 1000 + column * 7 + 3);
}

__device__ __forceinline__ uint32_t shared_address_raw(const void *pointer)
{
    // TMA 与 mbarrier 的 shared 操作数是 32-bit shared-memory offset；
    // CUDA C++ 传入的是 generic pointer，需要先转换 address space。
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64
//
// 指令名称：mov.u64 from %clock64
// 用途：读取 SM cycle counter，测量单次 load/store TMA 阶段的设备周期。
__device__ __forceinline__ uint64_t read_clock64_raw()
{
    uint64_t value = 0;
    asm volatile(
        "mov.u64 %0, %%clock64;\n"
        : "=l"(value)
        :
        : "memory");
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init
//
// 指令名称：mbarrier.init.shared::cta.b64
// 用途：初始化跟踪 TMA load 的 CTA transaction barrier。expected_arrivals=1
// 仅表示 thread 0 稍后执行一次 arrive，不表示 TMA 指令或字节数量。
__device__ __forceinline__ void mbarrier_init_raw(
    uint64_t *barrier,
    uint32_t expected_arrivals)
{
    const uint32_t barrier_address = shared_address_raw(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(barrier_address), "r"(expected_arrivals)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy-async
//
// 指令名称：fence.proxy.async.shared::cta
// 用途：把普通 shared proxy 完成的 mbarrier 初始化发布给 TMA async proxy。
__device__ __forceinline__ void fence_proxy_async_shared_cta_raw()
{
    asm volatile(
        "fence.proxy.async.shared::cta;\n"
        :
        :
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive
//
// 指令名称：mbarrier.arrive.expect_tx.shared::cta.b64
// 用途：让 thread 0 到达 barrier，并登记 TMA load 应完成的 tile 总字节数。
__device__ __forceinline__ void mbarrier_arrive_expect_tx_raw(
    uint64_t *barrier,
    uint32_t transaction_bytes)
{
    const uint32_t barrier_address = shared_address_raw(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(barrier_address), "r"(transaction_bytes)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor
//
// 指令名称：
// cp.async.bulk.tensor.2d.shared::cta.global.tile
//     .mbarrier::complete_tx::bytes
// 用途：按 Tensor Map 的 shape、stride、box 与 swizzle，从 global tensor
// 的 (x, y) 坐标加载一个二维 tile，并把完成字节数报告给 mbarrier。
__device__ __forceinline__ void tma_load_2d_raw(
    void *shared_destination,
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    uint64_t *barrier)
{
    const uint32_t destination_address =
        shared_address_raw(shared_destination);
    const uint32_t barrier_address = shared_address_raw(barrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(destination_address),
          "l"(tensor_map),
          "r"(coordinate_x),
          "r"(coordinate_y),
          "r"(barrier_address)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
//
// 指令名称：mbarrier.try_wait.parity.shared::cta.b64
// 用途：轮询当前 phase，直到 arrival count 与 transaction byte count
// 都归零。初始化后的首个 phase 使用 parity 0。
__device__ __forceinline__ void mbarrier_wait_parity_raw(
    uint64_t *barrier,
    uint32_t parity)
{
    const uint32_t barrier_address = shared_address_raw(barrier);
    uint32_t complete = 0;
    do
    {
        asm volatile(
            "{\n"
            " .reg .pred done;\n"
            " mbarrier.try_wait.parity.shared::cta.b64 "
            "done, [%1], %2;\n"
            " selp.b32 %0, 1, 0, done;\n"
            "}\n"
            : "=r"(complete)
            : "r"(barrier_address), "r"(parity)
            : "memory");
    }
    while (complete == 0);
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor
//
// 指令名称：cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
// 用途：按输出 Tensor Map 的同一种 swizzle 解释 shared tile，执行逆布局
// 变换并写回 row-major global tensor。store 通过 bulk group 而非 mbarrier
// 跟踪完成状态。
__device__ __forceinline__ void tma_store_2d_raw(
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    const void *shared_source)
{
    const uint32_t source_address = shared_address_raw(shared_source);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensor_map),
          "r"(coordinate_x),
          "r"(coordinate_y),
          "r"(source_address)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
//
// 指令名称：cp.async.bulk.commit_group / cp.async.bulk.wait_group.read 0
// 用途：提交 TMA store，并等到 async proxy 已读完 shared source，之后可以
// 安全复用 shared tile。`.read` 不单独承诺 global 目的端已经对 host 可见；
// kernel completion 与 cudaDeviceSynchronize 提供最终的 host 可见性。
__device__ __forceinline__ void tma_store_wait_read_raw()
{
    asm volatile(
        "cp.async.bulk.commit_group;\n"
        :
        :);
    asm volatile(
        "cp.async.bulk.wait_group.read 0;\n"
        :
        :
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval
//
// 指令名称：mbarrier.inval.shared::cta.b64
// 用途：所有线程结束本次 TMA phase 后销毁 shared-memory barrier 状态。
__device__ __forceinline__ void mbarrier_invalidate_raw(uint64_t *barrier)
{
    const uint32_t barrier_address = shared_address_raw(barrier);
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :
        : "r"(barrier_address)
        : "memory");
}

// ChunkCount 表示共享内存每行包含多少个 16B chunk：
//   2 / 4 / 8 chunk 分别对应 32B / 64B / 128B inner span。
// None 模式也使用 8 chunk，作为相同 128B tile 大小下的无 swizzle 基线。
template <int ChunkCount>
__global__ void tma_swizzle_kernel(
    __grid_constant__ const CUtensorMap input_tensor_map,
    __grid_constant__ const CUtensorMap output_tensor_map,
    uint64_t *completion_cycles)
{
    // 1024B 是 128B swizzle pattern 的重复边界，同时也是 32B、64B 模式
    // 256B、512B 重复边界的公倍数。这样三种模式的 swizzle base offset
    // 都是 0，避免共享内存起始地址额外引入行偏移，便于单独验证模式本身。
    // TMA 的最低共享内存对齐要求是 128B；这里的 1024B 是有意加强对齐。
    __shared__ __align__(1024) uint4 shared_tile[kRows][ChunkCount];

    // load 方向通过 mbarrier 跟踪 global-to-shared 异步事务。
    __shared__ __align__(8) uint64_t transaction_barrier;

    // 一次搬运覆盖完整的二维 shared tile。每个 uint4 是 16B，所以：
    // 32B mode = 8 * 2 * 16B = 256B，
    // 64B mode = 8 * 4 * 16B = 512B，
    // 128B/None mode = 8 * 8 * 16B = 1024B。
    constexpr uint32_t transaction_bytes =
        kRows * ChunkCount * sizeof(uint4);

    // 只由 thread 0 初始化和发起 bulk TMA。expected arrival count 为 1，
    // 对应稍后的唯一一次 arrive_expect_tx；它不是 TMA 事务数量。
    if (threadIdx.x == 0)
    {
        mbarrier_init_raw(&transaction_barrier, 1);

        // 将普通 shared proxy 完成的 barrier 初始化发布给 TMA async proxy。
        fence_proxy_async_shared_cta_raw();
    }

    // 确保初始化阶段结束后再进入 TMA load/store 阶段。
    __syncthreads();

    if (threadIdx.x == 0)
    {
        const uint64_t start_cycles = read_clock64_raw();

        // thread 0 到达 barrier，并登记 load 需要完成的 tile 总字节数。
        mbarrier_arrive_expect_tx_raw(
            &transaction_barrier,
            transaction_bytes);

        // 从 input tensor 的二维坐标 (x=0, y=0) 发起 TMA load。
        // 硬件先按 Tensor Map 的 global dimensions/stride 取数，再按该
        // Tensor Map 的 swizzle 模式把 16B chunks 排列到 shared_tile。
        tma_load_2d_raw(
            shared_tile,
            &input_tensor_map,
            0,
            0,
            &transaction_barrier);

        // 等待第一个 barrier phase 完成，保证 store 开始读取 shared_tile
        // 之前，global-to-shared 搬运和 swizzle 已经全部完成。
        mbarrier_wait_parity_raw(&transaction_barrier, 0);

        // output Tensor Map 使用与 input 相同的 shape 和 swizzle，只替换了
        // global-memory 基地址。TMA store 按同一 swizzle 规则解释共享内存，
        // 将布局反变换后写回普通 row-major global memory。
        tma_store_2d_raw(
            &output_tensor_map,
            0,
            0,
            shared_tile);

        // store 属于 bulk async group。这里提交 group，并等待异步代理读完
        // shared_tile，使共享内存可以安全复用。它不代表 global 写入已经
        // 对 host 可见；测试依靠 kernel 完成和 cudaDeviceSynchronize 验证。
        tma_store_wait_read_raw();

        // 该计数覆盖 TMA load 等待，以及 store 读完 shared source 的时间；
        // 它不是 global-memory store 最终可见延迟。
        completion_cycles[0] = read_clock64_raw() - start_cycles;
    }

    // 确保没有线程仍处于本次 TMA 阶段，再销毁 transaction barrier。
    __syncthreads();
    if (threadIdx.x == 0)
    {
        mbarrier_invalidate_raw(&transaction_barrier);
    }
}

CUtensorMap make_tensor_map(
    uint32_t *device_data,
    int tile_columns,
    CUtensorMapSwizzle swizzle)
{
    // 为同一个二维 row-major global tensor 编码 tiled Tensor Map：
    //
    //   global byte address(x, y)
    //     = base_address + y * (32 * sizeof(uint32_t))
    //                    + x * sizeof(uint32_t)
    //
    // Tensor Map 同时保存单次搬运的 box shape 和 shared-memory swizzle。
    // kernel 只传入描述符与 (x, y) 坐标，TMA 硬件负责地址生成及布局变换。

    // CUDA Driver 要求 CUtensorMap 输出对象按 64B 对齐。
    alignas(64) CUtensorMap tensor_map{};

    // Tensor Map 的 dimension 0 是连续的 x/column 维，dimension 1 是
    // y/row 维，因此 global tensor 的逻辑形状为 [32 columns, 8 rows]。
    const cuuint64_t global_dimensions[] = {
        kGlobalColumns,
        kRows
    };

    // rank-2 只需要一个 global stride：从第 y 行到第 y+1 行跨过
    // 32 个 uint32_t，即 128B。global stride 的单位是字节。
    const cuuint64_t global_strides[] = {
        kGlobalColumns * sizeof(uint32_t)
    };

    // 一次 TMA 指令读取 tile_columns 列和全部 kRows 行。inner dimension
    // 的字节数为 tile_columns * 4，必须不超过所选 swizzle 的 span：
    // 32B mode <= 32B，64B mode <= 64B，128B mode <= 128B。
    const cuuint32_t box_dimensions[] = {
        static_cast<cuuint32_t>(tile_columns),
        kRows
    };

    // 两个维度都逐元素遍历，不进行逻辑降采样。
    const cuuint32_t element_strides[] = {1, 1};

    check_driver(
        cuTensorMapEncodeTiled(
            &tensor_map,
            // coordinate 和 box dimension 都以 uint32_t 元素为单位。
            CU_TENSOR_MAP_DATA_TYPE_UINT32,
            // 使用 tensor.2d TMA 指令，坐标顺序为 (x, y)。
            2,
            device_data,
            global_dimensions,
            global_strides,
            box_dimensions,
            element_strides,
            // global tensor 是普通 row-major，不采用通道交错格式。
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            // swizzle 只改变 shared-memory 中 16B chunks 的排列；它不会
            // 改写 global-memory tensor 的 row-major 物理布局。
            swizzle,
            // 本测试不混入 L2 promotion 行为。
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            // 测试 box 完全位于 global tensor 内，不需要越界填充。
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");

    return tensor_map;
}

template <int ChunkCount>
void launch_swizzle(
    const CUtensorMap &input_tensor_map,
    const CUtensorMap &output_tensor_map,
    uint64_t *device_cycles)
{
    tma_swizzle_kernel<ChunkCount>
        <<<1, kBlockThreads>>>(
            input_tensor_map,
            output_tensor_map,
            device_cycles);
    check_cuda(cudaGetLastError(), "tma_swizzle_kernel launch");
}

template <int ChunkCount>
void run_mode(
    const char *name,
    CUtensorMapSwizzle swizzle,
    uint32_t *device_input,
    uint32_t *device_output,
    uint64_t *device_cycles,
    cudaEvent_t start_event,
    cudaEvent_t stop_event,
    const std::vector<uint32_t> &host_input)
{
    // ChunkCount 以 16B 为单位，换算为 uint32_t 列数。例如 32B mode
    // 使用 2 个 uint4 chunk，即每行搬运 8 个 uint32_t。
    constexpr int tile_columns = ChunkCount * kValuesPerChunk;

    // load/store 必须使用相同的 shape 和 swizzle，才能让 store 正确解释
    // load 生成的共享内存布局；两个 map 只在 global 基地址上不同。
    const CUtensorMap input_tensor_map =
        make_tensor_map(device_input, tile_columns, swizzle);
    const CUtensorMap output_tensor_map =
        make_tensor_map(device_output, tile_columns, swizzle);

    check_cuda(
        cudaMemset(device_output, 0, kInputBytes),
        "cudaMemset(device_output)");

    // 完成一次 input global -> swizzled shared -> output global 往返。
    launch_swizzle<ChunkCount>(
        input_tensor_map,
        output_tensor_map,
        device_cycles);
    check_cuda(cudaDeviceSynchronize(), "correctness synchronization");

    std::vector<uint32_t> host_output(kInputElements);
    uint64_t completion_cycles = 0;
    check_cuda(
        cudaMemcpy(
            host_output.data(),
            device_output,
            kInputBytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(output device-to-host)");
    check_cuda(
        cudaMemcpy(
            &completion_cycles,
            device_cycles,
            sizeof(completion_cycles),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(cycles device-to-host)");

    // 只验证本模式 box 覆盖的列。若 load/store 对共享内存 swizzle 布局的
    // 解释不一致，写回的 row/column 数据会发生置换并在这里被发现。
    // 因为 kernel 没有直接按物理下标读取 shared_tile，所以这里不单独
    // 断言每个 16B chunk 在共享内存中的实际位置。
    for (int row = 0; row < kRows; ++row)
    {
        for (int column = 0; column < tile_columns; ++column)
        {
            const uint32_t expected =
                host_input[row * kGlobalColumns + column];
            const uint32_t actual =
                host_output[row * kGlobalColumns + column];
            if (actual != expected)
            {
                throw std::runtime_error(
                    std::string(name) +
                    " verification failed at row=" +
                    std::to_string(row) +
                    ", column=" + std::to_string(column) +
                    ", expected=" + std::to_string(expected) +
                    ", actual=" + std::to_string(actual) +
                    ", row_chunk0=" +
                    std::to_string(
                        host_output[row * kGlobalColumns]) +
                    ", row_chunk1=" +
                    std::to_string(
                        host_output[
                            row * kGlobalColumns + kValuesPerChunk]));
            }
        }
    }

    for (int iteration = 0;
         iteration < kWarmupIterations;
         ++iteration)
    {
        launch_swizzle<ChunkCount>(
            input_tensor_map,
            output_tensor_map,
            device_cycles);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    check_cuda(
        cudaEventRecord(start_event),
        "cudaEventRecord(start_event)");
    for (int iteration = 0;
         iteration < kBenchmarkIterations;
         ++iteration)
    {
        launch_swizzle<ChunkCount>(
            input_tensor_map,
            output_tensor_map,
            device_cycles);
    }
    check_cuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord(stop_event)");
    check_cuda(
        cudaEventSynchronize(stop_event),
        "cudaEventSynchronize(stop_event)");

    float elapsed_ms = 0.0F;
    check_cuda(
        cudaEventElapsedTime(
            &elapsed_ms,
            start_event,
            stop_event),
        "cudaEventElapsedTime");
    const double average_microseconds =
        static_cast<double>(elapsed_ms) * 1000.0 /
        kBenchmarkIterations;

    std::cout << "  " << std::left << std::setw(10) << name
              << std::right << std::setw(8) << tile_columns * sizeof(uint32_t)
              << " B/row"
              << std::setw(12) << completion_cycles << " cycles"
              << std::setw(12) << std::fixed << std::setprecision(2)
              << average_microseconds << " us"
              << "  verified\n";
}

}  // namespace

int main()
{
    uint32_t *device_input = nullptr;
    uint32_t *device_output = nullptr;
    uint64_t *device_cycles = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::cout << "TMA shared-memory swizzle test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Global matrix" << kRows << " x "
                  << kGlobalColumns << " uint32\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Shared alignment" << 1024 << " bytes\n\n";

        if (properties.major < 9)
        {
            throw std::runtime_error(
                "TMA swizzle requires compute capability 9.0 or newer");
        }

        std::vector<uint32_t> host_input(kInputElements);
        for (int row = 0; row < kRows; ++row)
        {
            for (int column = 0; column < kGlobalColumns; ++column)
            {
                host_input[row * kGlobalColumns + column] =
                    make_input_value(row, column);
            }
        }

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_input),
                kInputBytes),
            "cudaMalloc(device_input)");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output),
                kInputBytes),
            "cudaMalloc(device_output)");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_cycles),
                sizeof(uint64_t)),
            "cudaMalloc(device_cycles)");
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                kInputBytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(input host-to-device)");
        check_cuda(
            cudaEventCreate(&start_event),
            "cudaEventCreate(start_event)");
        check_cuda(
            cudaEventCreate(&stop_event),
            "cudaEventCreate(stop_event)");

        std::cout << "[Modes]\n";
        std::cout << "  " << std::left << std::setw(10) << "Mode"
                  << std::right << std::setw(14) << "Inner span"
                  << std::setw(19) << "TMA completion"
                  << std::setw(15) << "Kernel"
                  << "  Result\n";

        // None 使用 128B/row 作为无布局变换基线；其余三项让 box 的 inner
        // dimension 恰好等于 swizzle span。该往返测试验证 Tensor Map 配置
        // 以及 load/store 对 swizzle 布局的解释一致；它不直接检查共享内存
        // 的物理排列，也不测量 bank conflict 收益。
        run_mode<8>(
            "None",
            CU_TENSOR_MAP_SWIZZLE_NONE,
            device_input,
            device_output,
            device_cycles,
            start_event,
            stop_event,
            host_input);
        run_mode<2>(
            "32B",
            CU_TENSOR_MAP_SWIZZLE_32B,
            device_input,
            device_output,
            device_cycles,
            start_event,
            stop_event,
            host_input);
        run_mode<4>(
            "64B",
            CU_TENSOR_MAP_SWIZZLE_64B,
            device_input,
            device_output,
            device_cycles,
            start_event,
            stop_event,
            host_input);
        run_mode<8>(
            "128B",
            CU_TENSOR_MAP_SWIZZLE_128B,
            device_input,
            device_output,
            device_cycles,
            start_event,
            stop_event,
            host_input);

        std::cout << "\n[SUCCESS] All TMA swizzle modes passed "
                  << "load/store round-trip verification\n";

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
        start_event = nullptr;
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
        stop_event = nullptr;
        check_cuda(cudaFree(device_cycles), "cudaFree(device_cycles)");
        device_cycles = nullptr;
        check_cuda(cudaFree(device_output), "cudaFree(device_output)");
        device_output = nullptr;
        check_cuda(cudaFree(device_input), "cudaFree(device_input)");
        device_input = nullptr;
    }
    catch (const std::exception &error)
    {
        if (start_event != nullptr)
        {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr)
        {
            cudaEventDestroy(stop_event);
        }
        if (device_cycles != nullptr)
        {
            cudaFree(device_cycles);
        }
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }
        if (device_input != nullptr)
        {
            cudaFree(device_input);
        }

        std::cerr << "[ERROR] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
