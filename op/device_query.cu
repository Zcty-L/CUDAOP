#include <iostream>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>

// CUDA error check macro (no printf)
#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess)                                         \
        {                                                               \
            std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__      \
                      << " " << cudaGetErrorString(err) << std::endl;   \
            exit(1);                                                    \
        }                                                               \
    } while (0)

static int get_attr(cudaDeviceAttr attr, int dev)
{
    int val = 0;
    cudaError_t err = cudaDeviceGetAttribute(&val, attr, dev);
    if (err != cudaSuccess)
    {
        return -1;
    }
    return val;
}

// -----------------------------------------------------------------------
//  Print helpers — aligned, no printf
// -----------------------------------------------------------------------
static void sep(const char *title)
{
    int tlen = (int)strlen(title);
    int dots = 55 - tlen;
    if (dots < 2)
    {
        dots = 2;
    }
    std::cout << "\n--- " << title << " " << std::string(dots, '-') << "\n";
}

static void p_str(const char *key, const char *val)
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  " << val << "\n";
}

static void p_int(const char *key, int val)
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  " << val << "\n";
}

template <typename T>
static void p_val(const char *key, T val)
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  " << val << "\n";
}

static void p_bytes(const char *key, size_t bytes)
{
    const char *unit[] = {"B", "KB", "MB", "GB", "TB"};
    int ui = 0;
    double d = (double)bytes;
    while (d >= 1024.0 && ui < 4)
    {
        d /= 1024.0;
        ui++;
    }
    if (ui == 0)
    {
        p_val(key, bytes);
    }
    else
    {
        std::cout << "  " << std::left << std::setw(36) << key
                  << "  " << std::fixed << std::setprecision(2) << d
                  << " " << unit[ui] << "  (" << bytes << " B)\n";
    }
}

static void p_bw(const char *key, double bw)
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  " << std::fixed << std::setprecision(2) << bw
              << " GB/s\n";
}

static void p_triple(const char *key, const int d[3])
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  (" << d[0] << ", " << d[1] << ", " << d[2] << ")\n";
}

static void p_flag(const char *key, bool ok)
{
    std::cout << "  " << std::left << std::setw(36) << key
              << "  " << (ok ? "Supported" : "Not supported") << "\n";
}

// -----------------------------------------------------------------------
//  Compute-capability based feature checks
// -----------------------------------------------------------------------
static bool has_tc(int maj, int min)
{
    return maj * 10 + min >= 70;
}
static bool has_wgmma(int maj, int min)
{
    return maj * 10 + min >= 90;
}
static bool has_acp(int maj, int min)
{
    return maj * 10 + min >= 80;
}
static bool has_dp4a(int maj, int min)
{
    return maj * 10 + min >= 61;
}

// -----------------------------------------------------------------------
//  SM core-count lookup
// -----------------------------------------------------------------------
static int cores_per_sm(int maj)
{
    if (maj >= 12 || maj == 9)
    {
        return 128;
    }
    if (maj == 8)
    {
        return 64;
    }
    if (maj == 7)
    {
        return 64;
    }
    return 128;
}

// -----------------------------------------------------------------------
//  Occupancy reference
// -----------------------------------------------------------------------
static void print_occupancy_table(int maxThdPerSM, int maxBlkPerSM,
                                   int maxThdPerBlk, int warpSz)
{
    sep("Occupancy Reference (common block sizes)");
    std::cout << "  Block Sz  | Blocks/SM | Warps/SM | Thds/SM\n";
    std::cout << "  ────────────────────────────────────────────\n";

    int sizes[] = {
        64, 128, 192, 256, 320, 384, 448, 512,
        576, 640, 704, 768, 832, 896, 960, 1024
    };
    for (int bs : sizes)
    {
        if (bs > maxThdPerBlk)
        {
            break;
        }
        int bThd = maxThdPerSM / bs;
        int blks = (bThd < maxBlkPerSM) ? bThd : maxBlkPerSM;
        if (blks == 0)
        {
            continue;
        }
        int warps = blks * (bs / warpSz);
        int thds  = blks * bs;
        std::cout << "  " << std::right << std::setw(7) << bs
                  << "    |  " << std::right << std::setw(6) << blks
                  << "   |  " << std::right << std::setw(6) << warps
                  << "   |  " << std::right << std::setw(6) << thds
                  << "\n";
    }
}

// -----------------------------------------------------------------------
int main()
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0)
    {
        std::cout << "No CUDA-capable devices found.\n";
        return 0;
    }

    for (int dev = 0; dev < devCount; dev++)
    {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

        int drvVer = 0, runVer = 0;
        CUDA_CHECK(cudaDriverGetVersion(&drvVer));
        CUDA_CHECK(cudaRuntimeGetVersion(&runVer));

        // Clock rates via device-attribute API (fields removed in CUDA 13)
        int gpuClk    = get_attr(cudaDevAttrClockRate, dev);       // kHz
        int memClk    = get_attr(cudaDevAttrMemoryClockRate, dev); // kHz
        int memBus    = p.memoryBusWidth;
        double bw     = 2.0 * (double)memClk * 1000.0 * (double)memBus / 8.0 / 1e9;
        bool black    = (p.major == 12);

        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════════\n"
                  << "║  CUDA Device #" << dev << "\n"
                  << "╚══════════════════════════════════════════════════════════════\n";

        std::cout << "  " << std::left << std::setw(36) << "CUDA Driver / Runtime"
                  << "  " << (drvVer / 1000) << "." << ((drvVer % 1000) / 10)
                  << "  /  "
                  << (runVer / 1000) << "." << ((runVer % 1000) / 10) << "\n";

        // ---- Identity ----
        sep("Device Identity");
        p_str("Name", p.name);
        p_int("Compute Capability", p.major * 10 + p.minor);
        p_str("Architecture", black   ? "Blackwell" :
              (p.major >= 9) ? "Hopper" :
              (p.major >= 8) ? "Ampere / Ada" :
              (p.major >= 7) ? "Volta / Turing" : "Pre-Volta");

        // ---- Memory Hierarchy ----
        sep("Memory Hierarchy");
        p_bytes("Global Memory", p.totalGlobalMem);
        p_bytes("L2 Cache", p.l2CacheSize);
        p_bytes("Shared Memory / Block", p.sharedMemPerBlock);
        p_bytes("Shared Memory / SM", p.sharedMemPerMultiprocessor);
        p_bytes("Shared Memory / Block (opt-in)", p.sharedMemPerBlockOptin);
        p_int("Registers / Block", p.regsPerBlock);
        p_int("Registers / SM", p.regsPerMultiprocessor);

        // ---- Memory Performance ----
        sep("Memory Performance");
        p_int("Memory Bus Width (bit)", memBus);
        p_int("Memory Clock (MHz)", memClk / 1000);
        p_int("GPU Clock (MHz)", gpuClk / 1000);
        p_bw("Estimated Bandwidth", bw);

        // ---- Roofline Analysis ----
        // Peak = SM * cores/SM * ops_per_cycle * clock(Hz)
        double base =
            (double)p.multiProcessorCount * (double)cores_per_sm(p.major)
            * (double)gpuClk * 1000.0;
        double peakFP32  = base * 2.0;              // FMA = 2 FLOPs/cycle
        double peakFP16  = base * 4.0;              // 2× FP32
        double peakINT8  = base * 8.0;              // 4× FP32

        double rFP32 = peakFP32 / 1e9 / bw;
        double rFP16 = peakFP16 / 1e9 / bw;
        double rINT8 = peakINT8 / 1e9 / bw;

        sep("Roofline Analysis (non-Tensor Core)");
        auto fmt_tflops = [](double val) -> std::string
        {
            return std::to_string(val / 1e12).substr(0, 5) + " TFLOPs";
        };
        auto fmt_ridge  = [](double val) -> std::string
        {
            return std::to_string(val).substr(0, 5) + " FLOPs/Byte";
        };

        p_val("Peak FP32", fmt_tflops(peakFP32));
        p_val("Peak FP16", fmt_tflops(peakFP16));
        p_val("Peak INT8", fmt_tflops(peakINT8));
        p_val("Ridge Point (FP32)", fmt_ridge(rFP32));
        p_val("Ridge Point (FP16)", fmt_ridge(rFP16));
        p_val("Ridge Point (INT8)", fmt_ridge(rINT8));
        std::cout << "\n"
                  << "  I >> ridge point → compute-bound\n"
                  << "  I << ridge point → memory-bound\n"
                  << "  (I = arithmetic intensity = FLOPs / Bytes accessed)\n"
                  << "  ⚡ Tensor Core (WGMMA) throughput is significantly\n"
                  << "    higher — ridge points shift right accordingly.\n";

        // ---- Core ----
        sep("Core Configuration");
        p_int("Multiprocessors (SM)", p.multiProcessorCount);
        p_int("CUDA Cores / SM (est.)", cores_per_sm(p.major));
        p_int("Total CUDA Cores (est.)",
              p.multiProcessorCount * cores_per_sm(p.major));
        p_int("Warp Size", p.warpSize);

        // ---- Occupancy ----
        sep("Occupancy Limits");
        p_int("Max Threads / Block", p.maxThreadsPerBlock);
        p_int("Max Threads / SM", p.maxThreadsPerMultiProcessor);
        p_int("Max Blocks / SM", p.maxBlocksPerMultiProcessor);
        p_int("Max Warps / SM", p.maxThreadsPerMultiProcessor / p.warpSize);
        p_triple("Max Block Dimensions", p.maxThreadsDim);
        p_triple("Max Grid Dimensions", p.maxGridSize);
        p_int("Async Engines", p.asyncEngineCount);

        print_occupancy_table(p.maxThreadsPerMultiProcessor,
                              p.maxBlocksPerMultiProcessor,
                              p.maxThreadsPerBlock, p.warpSize);

        // ---- Features ----
        sep("Feature Support");
        p_flag("Tensor Cores", has_tc(p.major, p.minor));
        p_flag("WGMMA (sm90+)", has_wgmma(p.major, p.minor));
        p_flag("Async Copy (sm80+)", has_acp(p.major, p.minor));
        p_flag("DP4A / Integer MMA (sm61+)", has_dp4a(p.major, p.minor));
        p_flag("Unified Addressing", p.unifiedAddressing > 0);
        p_flag("Concurrent Kernels", p.concurrentKernels > 0);
        p_flag("Managed Memory", p.managedMemory > 0);
        p_flag("Concurrent Managed Access", p.concurrentManagedAccess > 0);
        p_flag("Pageable Memory Access", p.pageableMemoryAccess > 0);
        p_flag("ECC Enabled", p.ECCEnabled > 0);
        p_flag("Can Map Host Memory", p.canMapHostMemory > 0);
        p_flag("Cooperative Launch", p.cooperativeLaunch > 0);
    }

    std::cout << "\n"
              << "  Tip: ./build/device_query | grep -i <keyword>\n"
              << "\n";
    return 0;
}
