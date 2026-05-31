# mul_u8 Kernel Design

## 1. Overview
A pointwise CUDA kernel located at `@op/mul/mul_u8.cu` to perform bitwise element-wise multiplication on the lowest bit of two `u8` tensors (`hx` and `xt`) of shape `[1, c, h, w]`. 

## 2. Architecture & Data Flow
- **Vectorized 1D Grid**: Flattens the tensor to a 1D array of size `N = c * h * w`. Each thread handles 4 elements.
- **Inputs**: 
  - `hx`, `xt`: `u8` arrays.
  - `t`: Integer representing the time step.
- **Outputs**:
  - `ht`: `u8` array. The kernel performs a Read-Modify-Write operation: `ht |= (((hx & 1) & (xt & 1)) << t)`.
  - `h_float`: Templated array (`float` or `half`). Stores the exact result of the multiplication as `1.0` or `0.0`.
- **Vector Types**: Employs `uchar4` for 4-byte loads/stores of the `u8` tensors, `float4` for `float`, and `half2` for `half` to maximize memory bandwidth.

## 3. Error Handling & Edge Cases
- **Remainder Handling**: Uses boundary checks (`if (idx * 4 + i < N)`) for arrays whose total element count `N` is not perfectly divisible by 4.

## 4. Testing
- Contains a CPU reference function (`mul_u8_cpu_ref`) replicating the logic for verification.
- A test function `test_mul_u8` generating random inputs, comparing kernel output against the CPU reference, and printing "PASS/FAIL".
- Includes CUDA events profiling to log the average execution time over 100 iterations.
