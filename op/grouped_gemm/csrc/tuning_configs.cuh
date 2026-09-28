#pragma once

#include "cutlass/gemm/gemm.h"

namespace cudaop::grouped_gemm
{

template <
    int ThreadblockM,
    int ThreadblockN,
    int ThreadblockK,
    int WarpM,
    int WarpN,
    int WarpK,
    int InstructionK,
    int Stages>
struct TuningShapeConfig
{
    static constexpr int kThreads =
        (ThreadblockM / WarpM) *
        (ThreadblockN / WarpN) *
        32;

    static_assert(ThreadblockM % WarpM == 0);
    static_assert(ThreadblockN % WarpN == 0);
    static_assert(ThreadblockK % WarpK == 0);
    static_assert(WarpK / InstructionK == 2);
    static_assert(
        ThreadblockM * ThreadblockK >= kThreads * 8,
        "A tile must provide at least one 128-bit access per thread");
    static_assert(
        ThreadblockN * ThreadblockK >= kThreads * 8,
        "B tile must provide at least one 128-bit access per thread");

    using ThreadblockShape = cutlass::gemm::GemmShape<
        ThreadblockM,
        ThreadblockN,
        ThreadblockK>;
    using WarpShape = cutlass::gemm::GemmShape<
        WarpM,
        WarpN,
        WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<
        16,
        8,
        InstructionK>;
    static constexpr int kStages = Stages;
};

}  // namespace cudaop::grouped_gemm
