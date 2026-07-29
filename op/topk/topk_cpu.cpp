#include "op/topk/topk.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cudaop
{
namespace
{

// 比较一行中的两个元素是否应按 Top-K 顺序排在前面。
// 排序规则与 CUDA 实现保持一致：
// 1. NaN 排在所有普通数值之前；
// 2. 普通数值按降序排列；
// 3. 数值相同时，较小的列索引优先，保证结果确定。
bool index_is_better(
    const float* row,
    int32_t left_index,
    int32_t right_index)
{
    const float left = row[left_index];
    const float right = row[right_index];
    const bool left_nan = std::isnan(left);
    const bool right_nan = std::isnan(right);

    // 只有一侧为 NaN 时，将 NaN 视为更大的元素。
    if (left_nan != right_nan)
    {
        return left_nan;
    }

    // return true; lhs 应排在 rhs 前面
    // return false; 不能判定 lhs 应排在 rhs 前面
    // 两侧均为普通数值时，较大的数值排在前面。
    if (!left_nan && left != right)
    {
        return left > right;
    }

    // 相同数值或两侧均为 NaN 时，通过索引建立确定的次序。
    return left_index < right_index;
}

}  // namespace

// 对行主序的二维 float32 矩阵逐行计算 Top-K。
//
// input   : [rows, columns] 输入矩阵。
// values  : [rows, k]，保存每行按降序排列的 Top-K 数值。
// indices : [rows, k]，保存对应数值在输入行中的列索引。
void topk_cpu(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k)
{
    if (input == nullptr || values == nullptr || indices == nullptr)
    {
        throw std::invalid_argument("Top-K pointers must not be null");
    }

    if (rows <= 0 || columns <= 0 || k <= 0 || k > columns)
    {
        throw std::invalid_argument("Invalid Top-K dimensions");
    }

    std::vector<int32_t> order(columns);
    for (int row_index = 0; row_index < rows; ++row_index)
    {
        const float* row = input + static_cast<size_t>(row_index) * columns;

        // 重置为当前行的完整列索引序列 [0, columns) [0, 1, 2, ..., columns-1]
        std::iota(order.begin(), order.end(), 0);

        // 只排序前 k 个索引，避免对整行执行完整排序。
        // 完成后 [order.begin(), order.begin() + k) 已按比较规则有序，
        // 后续索引的顺序未定义。复杂度约为 O(columns * log(k))。
        std::partial_sort(
            order.begin(),
            order.begin() + k,
            order.end(),
            [row](int32_t left, int32_t right)
            {
                return index_is_better(row, left, right);
            });

        // 将当前行的 Top-K 数值和原始列索引写入连续输出缓冲区。
        const size_t output_offset = static_cast<size_t>(row_index) * k;
        for (int rank = 0; rank < k; ++rank)
        {
            const int32_t selected_index = order[rank];
            values[output_offset + rank] = row[selected_index];
            indices[output_offset + rank] = selected_index;
        }
    }
}

}  // namespace cudaop
