---
layout: post
title: "Optimizing GPU Matrix Transpose: From 14% to 88% of Peak Bandwidth"
date: 2025-12-24
categories: gpu cuda
tags: [transpose, bandwidth, vectorization, shared-memory]
permalink: /gpu-matrix-transpose-optimization/
---

### Introduction

Matrix transpose is a fundamental operation in linear algebra and a canonical example for understanding GPU memory access patterns. A naive implementation achieves only 14% of theoretical peak bandwidth, while systematic optimization can reach 88%.

This post explores the optimization journey from naive transpose through shared memory tiling, bank conflict avoidance, thread coarsening, and vectorization. The progression demonstrates how GPU architecture features, including memory coalescing, shared memory banking, and wide vector loads, combine to achieve near-optimal performance.

This work builds on [Mark Harris's efficient matrix transpose tutorial](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/){:target="_blank" rel="noopener noreferrer"}, extending it with systematic exploration of thread coarsening factors, vectorization strategies (float2/float4), and larger tile sizes (64×64) on modern H100 architecture. The combination of coarsening with vectorization proves critical for achieving 87.7% of peak bandwidth.

The benchmark code is available in this [repository](https://github.com/yencal/gpu-matrix-transpose.git){:target="_blank" rel="noopener noreferrer"}.

### The Transpose Problem

Transposing a matrix swaps rows and columns: element (i,j) moves to position (j,i). The challenge is that global memory reads and writes cannot both be coalesced simultaneously.

**Naive approach:**

```cpp
output[x * N + y] = input[y * N + x];
```

This coalesces reads (consecutive threads access consecutive memory locations) but not writes (consecutive threads write to strided locations N elements apart). The result is poor memory bandwidth utilization.

### Implementation Progression

#### Naive Transpose

The baseline implementation performs direct indexing with no optimization:

```cpp
__global__ void TransposeNaive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        output[x * N + y] = input[y * N + x];
    }
}
```

This achieves only ~475 GB/s (14% of H100 peak) due to uncoalesced writes.

#### Tiled with Shared Memory

Using shared memory as an intermediate staging area allows both reads and writes to be coalesced:

```cpp
template <int TILE_DIM>
__global__ void TransposeBankConflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{ 
    __shared__ float smem[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        smem[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        output[y * N + x] = smem[threadIdx.x][threadIdx.y];
    }
}
```

The transpose is accomplished through two coordinate swaps:
1. Shared memory index swap: `smem[ty][tx]` → `smem[tx][ty]`
2. Block coordinate swap: `blockIdx.x, blockIdx.y` → `blockIdx.y, blockIdx.x`

This improves bandwidth to ~996 GB/s (30% of peak) but introduces 32-way shared memory bank conflicts during column access.

#### Eliminating Bank Conflicts

Shared memory is divided into 32 banks. When threads in a warp access the same bank, the requests serialize. Reading a column from `smem[TILE_DIM][TILE_DIM]` causes all 32 threads to hit the same bank.

Adding one element of padding breaks the conflict pattern:

```cpp
__shared__ float smem[TILE_DIM][TILE_DIM+1];  // +1 padding
```

This single change improves bandwidth to ~1693 GB/s (50% of peak).

For more in-depth discussion about [bank conflicts](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/){:target="_blank" rel="noopener noreferrer"} and alternative solutions like [swizzling patterns](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/){:target="_blank" rel="noopener noreferrer"}, I recommend Lei Mao's excellent blog posts.

#### Thread Coarsening

Modern GPUs benefit from having each thread process multiple elements. This amortizes index calculations and improves instruction-level parallelism:

```cpp
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void TransposeNoBankConflictsCoarsen(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    __shared__ float smem[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        smem[threadIdx.y+j][threadIdx.x] = input[(y+j)*N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[(y+j)*N + x] = smem[threadIdx.x][threadIdx.y+j];
    }
}
```

Using `BLOCK_ROWS=8` (each thread processes 4 rows) achieves ~2772 GB/s (83% of peak).

#### Vectorized Memory Access

Combining vectorization with coarsening provides the final performance boost. The vectorized kernel unpacks wide loads into shared memory, then gathers and repacks for wide stores:

```cpp
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void TransposeVec4Coarsen(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{   
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * (TILE_DIM/4) + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int width = N / 4;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        float4 data = reinterpret_cast<const float4*>(input)[(y+j) * width + x];
        tile[threadIdx.y + j][threadIdx.x * 4 + 0] = data.x;
        tile[threadIdx.y + j][threadIdx.x * 4 + 1] = data.y;
        tile[threadIdx.y + j][threadIdx.x * 4 + 2] = data.z;
        tile[threadIdx.y + j][threadIdx.x * 4 + 3] = data.w;
    }
    __syncthreads();

    x = blockIdx.y * (TILE_DIM/4) + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        float4 data;
        data.x = tile[threadIdx.x * 4 + 0][threadIdx.y + j];
        data.y = tile[threadIdx.x * 4 + 1][threadIdx.y + j];
        data.z = tile[threadIdx.x * 4 + 2][threadIdx.y + j];
        data.w = tile[threadIdx.x * 4 + 3][threadIdx.y + j];
        reinterpret_cast<float4*>(output)[(y+j) * width + x] = data;
    }
}
```

### Results and Discussion

The benchmark evaluates 20 kernel variants on H100 with a 32768×32768 matrix (8.6 GB read+write):

```
Kernel                               Block    Time(ms)      BW(GB/s)    % Peak
------------------------------------------------------------------------------
Naive                                32x32     18.0645         475.5     14.2%
Tiled                                32x32      8.6235         996.1     29.7%
TiledPadded                          32x32      5.0724        1693.4     50.5%
Coarsen2                             32x16      3.6014        2385.1     71.1%
Coarsen4                              32x8      3.0984        2772.4     82.7%
Coarsen8                              32x4      3.0951        2775.3     82.8%
Vec2                                 16x32      3.6041        2383.4     71.1%
Vec2Coarsen2                         16x16      3.0983        2772.5     82.7%
Vec2Coarsen4                          16x8      3.0953        2775.1     82.8%
Vec2Coarsen8                          16x4      3.0951        2775.4     82.8%
Vec4                                  8x32      3.0997        2771.2     82.7%
Vec4Coarsen2                          8x16      3.0966        2774.0     82.7%
Vec4Coarsen4                           8x8      3.0960        2774.5     82.8%
Vec4Coarsen8                           8x4      3.0972        2773.4     82.7%
Vec2_T64                             32x32      2.9910        2872.0     85.7%
Vec2_T64_Coarsen2                    32x16      2.9220        2939.8     87.7%
Vec2_T64_Coarsen4                     32x8      2.9345        2927.2     87.3%
Vec4_T64                             16x32      2.9228        2938.9     87.7%
Vec4_T64_Coarsen2                    16x16      2.9347        2927.0     87.3%
Vec4_T64_Coarsen4                     16x8      2.9213        2940.5     87.7%
------------------------------------------------------------------------------
Theoretical Peak: 3352.3 GB/s
Best Achieved: 2940.5 GB/s (87.7%)
```

**Key observations:**

The optimization progression shows clear inflection points. Shared memory tiling provides a 2.1x improvement over naive (30% vs 14%). Adding padding for bank conflict avoidance delivers another 1.7x (50% vs 30%). Thread coarsening jumps to 83%, and larger tiles with vectorization reach the final 88%.

### Conclusion

Achieving near-optimal transpose performance requires combining multiple techniques:

- **Shared memory tiling:** Stage data to enable coalesced reads and writes
- **Bank conflict avoidance:** Add +1 padding to eliminate conflicts for scalar kernels. Vectorized kernels (Vec2/Vec4) still exhibit 2-way and 4-way bank conflicts respectively.
- **Thread coarsening:** Each thread processes multiple rows (dominant optimization)
- **Vectorization:** Use float2/float4 for wide memory transactions
- **Larger tiles:** 64×64 tiles reduce per-tile overhead

The optimized transpose reaches 2940 GB/s on H100, achieving 87.7% of the 3.35 TB/s theoretical peak.

### References

* [An Efficient Matrix Transpose in CUDA C/C++ (Mark Harris)](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/){:target="_blank" rel="noopener noreferrer"}