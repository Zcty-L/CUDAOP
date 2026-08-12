#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace
{

constexpr int kRows = 16;
constexpr int kColumns = 32;
constexpr int kTileRows = 8;
constexpr int kTileColumns = 16;
constexpr int kBlockThreads = kTileRows * kTileColumns;
constexpr int kGridColumns = kColumns / kTileColumns;
constexpr int kGridRows = kRows / kTileRows;
constexpr int kTileCount = kGridColumns * kGridRows;
constexpr size_t kElementCount = static_cast<size_t>(kRows) * kColumns;
constexpr int kLoad1dTileElements = kBlockThreads;
constexpr int kLoad1dTileCount =
    static_cast<int>(kElementCount) / kLoad1dTileElements;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkIterations = 200;
constexpr uint32_t kTransformMask = 0x5a5aa5a5U;
constexpr size_t kTensorBytes = kElementCount * sizeof(uint32_t);
constexpr uint32_t kTileBytes =
    kTileRows * kTileColumns * sizeof(uint32_t);

static_assert(kRows % kTileRows == 0, "tile rows must divide tensor rows");
static_assert(
    kColumns % kTileColumns == 0,
    "tile columns must divide tensor columns");
static_assert(kBlockThreads <= 1024, "thread block is too large");
static_assert(kTileBytes % 16 == 0, "TMA tile must preserve 16B granularity");
static_assert(
    kElementCount % kLoad1dTileElements == 0,
    "1D TMA tiles must cover the whole tensor");
static_assert(
    kLoad1dTileCount == kTileCount,
    "1D and 2D paths share one cycle buffer");

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
    uint32_t value = static_cast<uint32_t>(row * kColumns + column + 1);
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

__device__ __forceinline__ uint32_t shared_address(const void *pointer)
{
    // CUDA C++ shared pointer 是 generic 64-bit 地址，而下列 PTX 指令的
    // shared 操作数要求 32-bit shared-memory offset。
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init
//
// 指令名称：mbarrier.init.shared::cta.b64
// 用途：在 CTA shared memory 中初始化 transaction barrier。
// expected_arrivals 是当前 phase 需要的线程 arrive 次数，不是 TMA 指令数，
// 也不是待传输字节数。本例只有 thread 0 调用 arrive_expect_tx，因此传 1。
__device__ __forceinline__ void mbarrier_init_raw(
    uint64_t *barrier,
    uint32_t expected_arrivals)
{
    const uint32_t barrier_address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(barrier_address), "r"(expected_arrivals)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive
//
// 指令名称：mbarrier.arrive.expect_tx.shared::cta.b64
// 用途：让调用线程到达 barrier，同时登记本 phase 还需由异步事务完成的
// transaction_bytes。TMA 每完成相应字节数就递减 pending transaction count。
__device__ __forceinline__ void mbarrier_arrive_expect_tx_raw(
    uint64_t *barrier,
    uint32_t transaction_bytes)
{
    const uint32_t barrier_address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(barrier_address), "r"(transaction_bytes)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
//
// 指令名称：mbarrier.try_wait.parity.shared::cta.b64
// 用途：轮询指定 parity 的 phase。返回 predicate 为 true 时，arrival count
// 和 transaction count 都已归零。本例只使用初始化后的第一个 phase，parity=0。
__device__ __forceinline__ void mbarrier_wait_parity_raw(
    uint64_t *barrier,
    uint32_t parity)
{
    const uint32_t barrier_address = shared_address(barrier);
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
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval
//
// 指令名称：mbarrier.inval.shared::cta.b64
// 用途：最后一个 phase 完成并且所有线程停止使用 barrier 后，销毁其状态。
__device__ __forceinline__ void mbarrier_invalidate_raw(uint64_t *barrier)
{
    const uint32_t barrier_address = shared_address(barrier);
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :
        : "r"(barrier_address)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy-async
//
// 指令名称：fence.proxy.async.shared::cta
// 用途：在 generic shared-memory proxy 与 TMA async proxy 之间建立可见性。
// 本例在初始化 mbarrier 后使用一次，在普通线程改写 shared tile 后再使用一次。
__device__ __forceinline__ void fence_proxy_async_shared_cta_raw()
{
    asm volatile(
        "fence.proxy.async.shared::cta;\n"
        :
        :
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor
//
// 指令名称：
// cp.async.bulk.tensor.1d.shared::cta.global.tile
//     .mbarrier::complete_tx::bytes
// 用途：按 rank-1 Tensor Map 从 coordinate 起加载一个一维 box。coordinate
// 单位由 Tensor Map 的元素类型决定，本例是 uint32_t 元素而不是 byte。
// 这条路径由原 tma_copy.cu 合并而来，保留 1D TMA 指令的独立观察能力。
__device__ __forceinline__ void tma_load_1d_raw(
    void *shared_destination,
    const void *tensor_map,
    int32_t coordinate,
    uint64_t *barrier)
{
    const uint32_t destination_address = shared_address(shared_destination);
    const uint32_t barrier_address = shared_address(barrier);

    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2}], [%3];\n"
        :
        : "r"(destination_address),
          "l"(tensor_map),
          "r"(coordinate),
          "r"(barrier_address)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor
//
// 指令名称：
// cp.async.bulk.tensor.2d.shared::cta.global.tile
//     .mbarrier::complete_tx::bytes
// 用途：由 Tensor Map 描述 global tensor，并从 coordinate=(x, y) 开始把
// 一个二维 box 异步搬到 CTA shared memory。它不需要每个线程计算地址；
// 一个被选中的线程发出一次 bulk 指令即可。完成字节数报告给 mbarrier。
__device__ __forceinline__ void tma_load_2d_raw(
    void *shared_destination,
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    uint64_t *barrier)
{
    const uint32_t destination_address = shared_address(shared_destination);
    const uint32_t barrier_address = shared_address(barrier);

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
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor
//
// 指令名称：cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
// 用途：按输出 Tensor Map 和二维坐标解释 shared tile，并异步写入 global
// tensor。store 不使用 mbarrier，而是加入当前发出线程的 bulk async-group。
__device__ __forceinline__ void tma_store_2d_raw(
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    const void *shared_source)
{
    const uint32_t source_address = shared_address(shared_source);

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
// 指令名称：cp.async.bulk.commit_group / cp.async.bulk.wait_group 0
// 用途：提交当前线程尚未提交的 TMA store，并等待之前提交的所有 bulk
// async-group 完整结束。立即数 0 表示允许仍未完成的 prior group 数为 0，
// 不是“等待第 0 组”。这里不用 `.read`，因此也等待写目的端的动作完成。
__device__ __forceinline__ void tma_store_commit_and_wait_raw()
{
    asm volatile(
        "cp.async.bulk.commit_group;\n"
        :
        :);
    asm volatile(
        "cp.async.bulk.wait_group 0;\n"
        :
        :
        : "memory");
}

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

__global__ void ptx_tma_load_1d_kernel(
    __grid_constant__ const CUtensorMap input_tensor_map,
    uint32_t *output,
    uint64_t *completion_cycles)
{
    __shared__ __align__(128) uint32_t shared_tile[kLoad1dTileElements];
    __shared__ __align__(8) uint64_t transaction_barrier;

    if (threadIdx.x == 0)
    {
        mbarrier_init_raw(&transaction_barrier, 1);
        fence_proxy_async_shared_cta_raw();
    }
    __syncthreads();

    const int32_t coordinate =
        static_cast<int32_t>(blockIdx.x) * kLoad1dTileElements;
    if (threadIdx.x == 0)
    {
        const uint64_t start_cycles = read_clock64_raw();

        // 一次 1D load 搬运 128 个 uint32_t，即 512B。arrive count 与
        // transaction byte count 是 mbarrier 的两个独立完成条件。
        mbarrier_arrive_expect_tx_raw(&transaction_barrier, kTileBytes);
        tma_load_1d_raw(
            shared_tile,
            &input_tensor_map,
            coordinate,
            &transaction_barrier);
        mbarrier_wait_parity_raw(&transaction_barrier, 0);

        completion_cycles[blockIdx.x] =
            read_clock64_raw() - start_cycles;
    }

    // thread 0 等完 TMA 后，用 CTA barrier 将数据就绪状态发布给全部线程。
    // 每个线程从 shared tile 取一个元素，以普通 store 写回 global output。
    __syncthreads();
    const size_t output_index =
        static_cast<size_t>(coordinate) + threadIdx.x;
    output[output_index] = shared_tile[threadIdx.x];

    __syncthreads();
    if (threadIdx.x == 0)
    {
        mbarrier_invalidate_raw(&transaction_barrier);
    }
}

__global__ void ptx_tma_2d_kernel(
    __grid_constant__ const CUtensorMap input_tensor_map,
    __grid_constant__ const CUtensorMap output_tensor_map,
    uint64_t *completion_cycles)
{
    // TMA shared destination 至少保持 128B 对齐。本例无 swizzle，shared
    // tile 的物理布局就是 8 x 16 的 row-major uint32_t，合计 512B。
    __shared__ __align__(128) uint32_t shared_tile[kTileRows][kTileColumns];
    __shared__ __align__(8) uint64_t transaction_barrier;

    if (threadIdx.x == 0)
    {
        // expected_arrivals=1 对应下面 thread 0 唯一一次 arrive_expect_tx。
        mbarrier_init_raw(&transaction_barrier, 1);

        // mbarrier.init 由普通 shared proxy 写状态，TMA 通过 async proxy
        // 更新状态；发起 load 前必须显式建立两个 proxy 之间的可见性。
        fence_proxy_async_shared_cta_raw();
    }

    __syncthreads();

    const int32_t coordinate_x =
        static_cast<int32_t>(blockIdx.x) * kTileColumns;
    const int32_t coordinate_y =
        static_cast<int32_t>(blockIdx.y) * kTileRows;
    uint64_t start_cycles = 0;

    if (threadIdx.x == 0)
    {
        start_cycles = read_clock64_raw();

        // transaction byte count 是这条 TMA load 实际生成的整个 box 大小：
        // 16 columns * 8 rows * 4 B = 512 B。
        mbarrier_arrive_expect_tx_raw(&transaction_barrier, kTileBytes);

        // Tensor Map 保存 base、shape、stride、box 和布局；kernel 只补充本
        // CTA 的逻辑起点。coordinate 单位是 uint32_t 元素，不是字节。
        tma_load_2d_raw(
            shared_tile,
            &input_tensor_map,
            coordinate_x,
            coordinate_y,
            &transaction_barrier);

        // 初始化后的第一个 mbarrier phase 使用 parity 0。返回后，TMA 对
        // shared_tile 的 512B async-proxy 写入已经完成。
        mbarrier_wait_parity_raw(&transaction_barrier, 0);
    }

    // mbarrier wait 只由 thread 0 执行；CTA barrier 把“tile 已就绪”发布
    // 给其余线程。128 个线程恰好一线程处理一个 uint32_t。
    __syncthreads();

    const int local_index = static_cast<int>(threadIdx.x);
    const int local_row = local_index / kTileColumns;
    const int local_column = local_index % kTileColumns;
    shared_tile[local_row][local_column] ^= kTransformMask;

    // 先收齐所有普通线程对 shared_tile 的写入，再由 thread 0 把这些
    // generic-proxy 写入发布给即将读取 shared memory 的 TMA async proxy。
    __syncthreads();

    if (threadIdx.x == 0)
    {
        fence_proxy_async_shared_cta_raw();

        tma_store_2d_raw(
            &output_tensor_map,
            coordinate_x,
            coordinate_y,
            shared_tile);
        tma_store_commit_and_wait_raw();

        const int tile_index =
            static_cast<int>(blockIdx.y) * kGridColumns +
            static_cast<int>(blockIdx.x);
        completion_cycles[tile_index] = read_clock64_raw() - start_cycles;
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        mbarrier_invalidate_raw(&transaction_barrier);
    }
}

CUtensorMap make_tensor_map_1d(uint32_t *device_data)
{
    alignas(64) CUtensorMap tensor_map{};

    // rank-1 Tensor Map 把整个 16 x 32 matrix 视为 512 个连续 uint32_t。
    // rank-1 没有更高维 global stride；仍传入一个有效占位数组，避免依赖
    // Driver 对 nullptr 的版本差异。
    const cuuint64_t global_dimensions[] = {
        static_cast<cuuint64_t>(kElementCount)};
    const cuuint64_t global_strides[] = {0};
    const cuuint32_t box_dimensions[] = {
        static_cast<cuuint32_t>(kLoad1dTileElements)};
    const cuuint32_t element_strides[] = {1};

    check_driver(
        cuTensorMapEncodeTiled(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_UINT32,
            1,
            device_data,
            global_dimensions,
            global_strides,
            box_dimensions,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled 1D");

    return tensor_map;
}

CUtensorMap make_tensor_map_2d(uint32_t *device_data)
{
    // cuTensorMapEncodeTiled 在 host 端把地址生成所需元数据编码成不透明、
    // 64B 对齐的描述符。TMA 指令读取描述符，不会在 PTX 操作数中再次接收
    // global row stride 或 box shape。
    alignas(64) CUtensorMap tensor_map{};

    // Tensor Map 维度按“最内层到最外层”排列：dimension 0 是 x/column，
    // dimension 1 是 y/row，所以逻辑 shape 为 [32 columns, 16 rows]。
    const cuuint64_t global_dimensions[] = {
        static_cast<cuuint64_t>(kColumns),
        static_cast<cuuint64_t>(kRows)};

    // rank-2 只提供 dimension 1 的 byte stride：从一行跨到下一行需要
    // 32 * sizeof(uint32_t) = 128B。dimension 0 默认按元素连续。
    const cuuint64_t global_strides[] = {
        static_cast<cuuint64_t>(kColumns * sizeof(uint32_t))};

    // 每条 cp.async.bulk.tensor.2d 搬运一个 [16 columns, 8 rows] box。
    // box dimension 与 kernel coordinate 都以 descriptor 元素类型为单位。
    const cuuint32_t box_dimensions[] = {
        static_cast<cuuint32_t>(kTileColumns),
        static_cast<cuuint32_t>(kTileRows)};
    const cuuint32_t element_strides[] = {1, 1};

    check_driver(
        cuTensorMapEncodeTiled(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_UINT32,
            2,
            device_data,
            global_dimensions,
            global_strides,
            box_dimensions,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled 2D");

    return tensor_map;
}

void launch_tma_1d(
    const CUtensorMap &input_tensor_map,
    uint32_t *output,
    uint64_t *completion_cycles)
{
    ptx_tma_load_1d_kernel<<<kLoad1dTileCount, kBlockThreads>>>(
        input_tensor_map,
        output,
        completion_cycles);
    check_cuda(cudaGetLastError(), "ptx_tma_load_1d_kernel launch");
}

void launch_tma_2d(
    const CUtensorMap &input_tensor_map,
    const CUtensorMap &output_tensor_map,
    uint64_t *completion_cycles)
{
    const dim3 grid(kGridColumns, kGridRows);
    ptx_tma_2d_kernel<<<grid, kBlockThreads>>>(
        input_tensor_map,
        output_tensor_map,
        completion_cycles);
    check_cuda(cudaGetLastError(), "ptx_tma_2d_kernel launch");
}

size_t verify_output(
    const std::vector<uint32_t> &input,
    const std::vector<uint32_t> &output,
    uint32_t xor_mask,
    const char *path_name)
{
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;

    for (size_t index = 0; index < input.size(); ++index)
    {
        const uint32_t expected = input[index] ^ xor_mask;
        if (output[index] != expected)
        {
            if (mismatch_count == 0)
            {
                first_mismatch = index;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0)
    {
        throw std::runtime_error(
            std::string(path_name) + " verification failed: mismatches=" +
            std::to_string(mismatch_count) +
            ", first_index=" + std::to_string(first_mismatch) +
            ", expected=" +
            std::to_string(input[first_mismatch] ^ xor_mask) +
            ", actual=" + std::to_string(output[first_mismatch]));
    }

    return mismatch_count;
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

        std::cout << "\n[配置]\n"
                  << "  " << std::left << std::setw(32)
                  << "GPU" << properties.name << '\n'
                  << "  " << std::left << std::setw(32)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n'
                  << "  " << std::left << std::setw(32)
                  << "Global tensor" << kRows << " x " << kColumns
                  << " uint32\n"
                  << "  " << std::left << std::setw(32)
                  << "1D TMA box" << kLoad1dTileElements
                  << " uint32 = " << kTileBytes << " B\n"
                  << "  " << std::left << std::setw(32)
                  << "2D TMA box" << kTileRows << " x " << kTileColumns
                  << " = " << kTileBytes << " B\n"
                  << "  " << std::left << std::setw(32)
                  << "1D / 2D grid" << kLoad1dTileCount << " / "
                  << kGridColumns << " x " << kGridRows << '\n'
                  << "  " << std::left << std::setw(32)
                  << "Block threads" << kBlockThreads << '\n'
                  << "  " << std::left << std::setw(32)
                  << "Transform XOR mask" << "0x" << std::hex
                  << kTransformMask << std::dec << '\n'
                  << "  " << std::left << std::setw(32)
                  << "Warmup / iterations" << kWarmupIterations << " / "
                  << kBenchmarkIterations << '\n';

        if (properties.major < 9)
        {
            throw std::runtime_error(
                "cp.async.bulk.tensor requires compute capability 9.0+");
        }

        std::cout << "\n[阶段 1] 编码输入/输出 Tensor Map 并上传数据\n";
        std::vector<uint32_t> host_input(kElementCount);
        std::vector<uint32_t> host_output(kElementCount, 0);
        std::vector<uint64_t> host_cycles(kTileCount, 0);

        for (int row = 0; row < kRows; ++row)
        {
            for (int column = 0; column < kColumns; ++column)
            {
                host_input[row * kColumns + column] =
                    make_input_value(row, column);
            }
        }

        check_cuda(
            cudaMalloc(reinterpret_cast<void **>(&device_input), kTensorBytes),
            "cudaMalloc input");
        check_cuda(
            cudaMalloc(reinterpret_cast<void **>(&device_output), kTensorBytes),
            "cudaMalloc output");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_cycles),
                kTileCount * sizeof(uint64_t)),
            "cudaMalloc cycles");
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                kTensorBytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy input");
        check_cuda(cudaMemset(device_output, 0, kTensorBytes), "cudaMemset output");

        const CUtensorMap input_tensor_map_1d =
            make_tensor_map_1d(device_input);
        const CUtensorMap input_tensor_map_2d =
            make_tensor_map_2d(device_input);
        const CUtensorMap output_tensor_map_2d =
            make_tensor_map_2d(device_output);
        std::cout << "  一个 rank-1 与两个 rank-2 Tensor Map 编码成功\n";

        std::cout << "\n[阶段 2] 验证 1D TMA load → SMEM → 普通 GMEM store\n";
        launch_tma_1d(
            input_tensor_map_1d,
            device_output,
            device_cycles);
        check_cuda(cudaDeviceSynchronize(), "1D correctness synchronization");
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                kTensorBytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy 1D output");
        check_cuda(
            cudaMemcpy(
                host_cycles.data(),
                device_cycles,
                kTileCount * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy 1D cycles");

        const size_t mismatch_1d =
            verify_output(host_input, host_output, 0, "1D TMA load");
        const auto minimum_cycle_1d =
            *std::min_element(host_cycles.begin(), host_cycles.end());
        const auto maximum_cycle_1d =
            *std::max_element(host_cycles.begin(), host_cycles.end());
        std::cout << "  校验元素数                     : "
                  << kElementCount << '\n'
                  << "  mismatch                       : "
                  << mismatch_1d << '\n'
                  << "  1D TMA load cycles min/max     : "
                  << minimum_cycle_1d << " / " << maximum_cycle_1d << '\n';

        check_cuda(cudaMemset(device_output, 0, kTensorBytes), "reset output");
        std::cout << "\n[阶段 3] 验证 2D TMA load → SMEM XOR → TMA store\n";
        launch_tma_2d(
            input_tensor_map_2d,
            output_tensor_map_2d,
            device_cycles);
        check_cuda(cudaDeviceSynchronize(), "2D correctness synchronization");
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                kTensorBytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy 2D output");
        check_cuda(
            cudaMemcpy(
                host_cycles.data(),
                device_cycles,
                kTileCount * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy 2D cycles");

        const size_t mismatch_2d = verify_output(
            host_input,
            host_output,
            kTransformMask,
            "2D TMA round-trip");
        const auto minimum_cycle_2d =
            *std::min_element(host_cycles.begin(), host_cycles.end());
        const auto maximum_cycle_2d =
            *std::max_element(host_cycles.begin(), host_cycles.end());
        std::cout
                  << "  校验元素数                     : "
                  << kElementCount << '\n'
                  << "  mismatch                       : "
                  << mismatch_2d << '\n'
                  << "  2D TMA round-trip cycles       : "
                  << minimum_cycle_2d << " / " << maximum_cycle_2d << '\n';

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");

        std::cout << "\n[阶段 4] 测量 1D 与 2D kernel 延迟\n";
        for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
        {
            launch_tma_1d(
                input_tensor_map_1d,
                device_output,
                device_cycles);
        }
        check_cuda(cudaDeviceSynchronize(), "1D warmup synchronization");

        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");
        for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration)
        {
            launch_tma_1d(
                input_tensor_map_1d,
                device_output,
                device_cycles);
        }
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");

        float elapsed_1d_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&elapsed_1d_ms, start_event, stop_event),
            "cudaEventElapsedTime 1D");
        const double average_1d_microseconds =
            static_cast<double>(elapsed_1d_ms) * 1000.0 /
            kBenchmarkIterations;

        for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
        {
            launch_tma_2d(
                input_tensor_map_2d,
                output_tensor_map_2d,
                device_cycles);
        }
        check_cuda(cudaDeviceSynchronize(), "2D warmup synchronization");

        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start 2D");
        for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration)
        {
            launch_tma_2d(
                input_tensor_map_2d,
                output_tensor_map_2d,
                device_cycles);
        }
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop 2D");
        check_cuda(
            cudaEventSynchronize(stop_event),
            "cudaEventSynchronize stop 2D");

        float elapsed_2d_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&elapsed_2d_ms, start_event, stop_event),
            "cudaEventElapsedTime 2D");
        const double average_2d_microseconds =
            static_cast<double>(elapsed_2d_ms) * 1000.0 /
            kBenchmarkIterations;

        std::cout << "  平均 1D load kernel 延迟       : "
                  << std::fixed << std::setprecision(3)
                  << average_1d_microseconds << " us\n"
                  << "  平均 2D round-trip kernel 延迟 : "
                  << std::fixed << std::setprecision(3)
                  << average_2d_microseconds << " us\n";

        std::cout << "\n[SUCCESS] raw PTX 1D/2D TMA、mbarrier 与 "
                  << "Tensor Map 验证通过\n";

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
        start_event = nullptr;
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        stop_event = nullptr;
        check_cuda(cudaFree(device_cycles), "cudaFree cycles");
        device_cycles = nullptr;
        check_cuda(cudaFree(device_output), "cudaFree output");
        device_output = nullptr;
        check_cuda(cudaFree(device_input), "cudaFree input");
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

        std::cerr << "[FAILED] " << error.what() << '\n';
        return 1;
    }

    return 0;
}
