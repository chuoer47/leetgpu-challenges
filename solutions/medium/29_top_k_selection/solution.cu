#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

constexpr int TILE = 256;
constexpr int RADIX_BITS = 8;
constexpr int RADIX_SIZE = 1 << RADIX_BITS;   // 256

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            return;                                                             \
        }                                                                       \
    } while (0)

__device__ __forceinline__ unsigned int float_to_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

// ------------------------------------------------------------------
// 1) 多位直方图：一次扫描统计候选集合在 [shift, shift+7] 位上的分布
// ------------------------------------------------------------------
__global__ void digit_histogram_kernel(const float* __restrict__ input, int N,
                                       unsigned int mask, unsigned int prefix,
                                       int shift, unsigned int* __restrict__ hist) {
    __shared__ unsigned int s_hist[RADIX_SIZE];
    for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const unsigned int digit_mask = ((unsigned int)(RADIX_SIZE - 1)) << shift;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        unsigned int key = float_to_key(input[i]);
        if ((key & mask) == prefix) {
            unsigned int digit = (key & digit_mask) >> shift;
            atomicAdd(&s_hist[digit], 1u);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) {
        if (s_hist[i] > 0) atomicAdd(&hist[i], s_hist[i]);
    }
}

// ------------------------------------------------------------------
// 2) 每 block 统计 key > threshold 和 key == threshold 的个数
// ------------------------------------------------------------------
__global__ void count_gt_eq_kernel(const float* __restrict__ input, int N,
                                   unsigned int threshold,
                                   unsigned int* __restrict__ block_gt,
                                   unsigned int* __restrict__ block_eq) {
    unsigned int local_gt = 0, local_eq = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        unsigned int key = float_to_key(input[i]);
        if (key > threshold) local_gt++;
        else if (key == threshold) local_eq++;
    }

    __shared__ unsigned int s_gt[TILE];
    __shared__ unsigned int s_eq[TILE];
    int tid = threadIdx.x;
    s_gt[tid] = local_gt;
    s_eq[tid] = local_eq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_gt[tid] += s_gt[tid + s];
            s_eq[tid] += s_eq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_gt[blockIdx.x] = s_gt[0];
        block_eq[blockIdx.x] = s_eq[0];
    }
}

// ------------------------------------------------------------------
// 3) 收集：block 内先做 scan，然后用预先算好的 block base offset 写出
// ------------------------------------------------------------------
__global__ void collect_kernel(const float* __restrict__ input, int N,
                               unsigned int threshold,
                               const unsigned int* __restrict__ gt_base,
                               const unsigned int* __restrict__ eq_base,
                               const unsigned int* __restrict__ eq_accept,
                               float* __restrict__ output) {
    __shared__ unsigned int s_gt[TILE];
    __shared__ unsigned int s_eq[TILE];
    int tid = threadIdx.x;

    // 第一遍：统计本线程负责的元素
    unsigned int local_gt = 0, local_eq = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        unsigned int key = float_to_key(input[i]);
        if (key > threshold) local_gt++;
        else if (key == threshold) local_eq++;
    }
    s_gt[tid] = local_gt;
    s_eq[tid] = local_eq;
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int s = 1; s < TILE; s <<= 1) {
        unsigned int v_gt = (tid >= s) ? s_gt[tid - s] : 0u;
        unsigned int v_eq = (tid >= s) ? s_eq[tid - s] : 0u;
        __syncthreads();
        s_gt[tid] += v_gt;
        s_eq[tid] += v_eq;
        __syncthreads();
    }

    unsigned int gt_thread_off = s_gt[tid] - local_gt;   // exclusive offset
    unsigned int eq_thread_off = s_eq[tid] - local_eq;

    unsigned int gt_start = gt_base[blockIdx.x];
    unsigned int eq_start = eq_base[blockIdx.x];
    unsigned int eq_limit = eq_accept[blockIdx.x];

    // 第二遍：写出
    unsigned int wrote_gt = 0, wrote_eq = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        float val = input[i];
        unsigned int key = float_to_key(val);
        if (key > threshold) {
            output[gt_start + gt_thread_off + wrote_gt] = val;
            wrote_gt++;
        } else if (key == threshold) {
            unsigned int idx = eq_thread_off + wrote_eq;
            if (idx < eq_limit) {
                output[eq_start + idx] = val;
            }
            wrote_eq++;
        }
    }
}

// ------------------------------------------------------------------
// 4) 用 -inf 填充到 2 的幂长度（这样降序排序后哨兵落在末尾）
// ------------------------------------------------------------------
__global__ void fill_neg_inf_kernel(float* __restrict__ data, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = __int_as_float(0xff800000);   // -inf
    }
}

// ------------------------------------------------------------------
// 5) Bitonic Sort 单步 —— 直接降序
// ------------------------------------------------------------------
__global__ void bitonic_sort_step_desc(float* __restrict__ data, int m, int j, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        int ij = i ^ j;
        if (i < ij) {
            bool descending = ((i & size) == 0);
            unsigned int key_i  = float_to_key(data[i]);
            unsigned int key_ij = float_to_key(data[ij]);

            if (descending) {
                if (key_i < key_ij) {
                    float tmp = data[i];
                    data[i] = data[ij];
                    data[ij] = tmp;
                }
            } else {
                if (key_i > key_ij) {
                    float tmp = data[i];
                    data[i] = data[ij];
                    data[ij] = tmp;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// solve
// ------------------------------------------------------------------
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) return;
    if (k > N) k = N;

    int threads = TILE;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    // ---- 一次性分配所有设备缓冲区 ----
    unsigned int *d_hist = nullptr;
    unsigned int *d_block_gt = nullptr, *d_block_eq = nullptr;
    unsigned int *d_gt_base = nullptr, *d_eq_base = nullptr, *d_eq_accept = nullptr;
    float* d_sort = nullptr;

    CUDA_CHECK(cudaMalloc(&d_hist, RADIX_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_block_gt, blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_block_eq, blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_gt_base, blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_eq_base, blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_eq_accept, blocks * sizeof(unsigned int)));

    std::vector<unsigned int> h_hist(RADIX_SIZE);
    std::vector<unsigned int> h_block_gt(blocks), h_block_eq(blocks);
    std::vector<unsigned int> h_gt_base(blocks), h_eq_base(blocks), h_eq_accept(blocks);

    // ---- 多位 MSB-first radix select ----
    unsigned int mask = 0;
    unsigned int prefix = 0;
    int remaining_k = k;

    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        CUDA_CHECK(cudaMemset(d_hist, 0, RADIX_SIZE * sizeof(unsigned int)));
        digit_histogram_kernel<<<blocks, threads>>>(input, N, mask, prefix, shift, d_hist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, RADIX_SIZE * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        unsigned int acc = 0;
        int chosen_digit = 0;
        for (int d = RADIX_SIZE - 1; d >= 0; --d) {
            if (acc + h_hist[d] >= (unsigned int)remaining_k) {
                chosen_digit = d;
                break;
            }
            acc += h_hist[d];
        }
        remaining_k -= (int)acc;
        prefix |= ((unsigned int)chosen_digit << shift);
        mask   |= (((unsigned int)(RADIX_SIZE - 1)) << shift);
    }

    unsigned int threshold = prefix;   // 第 k 大的 key

    // ---- 每 block 统计 gt / eq 数量 ----
    count_gt_eq_kernel<<<blocks, threads>>>(input, N, threshold, d_block_gt, d_block_eq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_block_gt.data(), d_block_gt, blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_eq.data(), d_block_eq, blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // ---- host 端 prefix sum：得到每 block 的 base offset ----
    unsigned int total_gt = 0;
    for (int b = 0; b < blocks; ++b) total_gt += h_block_gt[b];
    int need_eq = k - (int)total_gt;
    if (need_eq < 0) need_eq = 0;

    unsigned int gt_run = 0, eq_run = 0;
    for (int b = 0; b < blocks; ++b) {
        h_gt_base[b] = gt_run;
        gt_run += h_block_gt[b];

        unsigned int eq_before = eq_run;
        unsigned int eq_base_local =
            (eq_before < (unsigned int)need_eq) ? eq_before : (unsigned int)need_eq;
        h_eq_base[b] = total_gt + eq_base_local;

        unsigned int accepted = 0;
        if ((unsigned int)need_eq > eq_before) {
            unsigned int cap = (unsigned int)need_eq - eq_before;
            accepted = (h_block_eq[b] < cap) ? h_block_eq[b] : cap;
        }
        h_eq_accept[b] = accepted;
        eq_run += h_block_eq[b];
    }

    CUDA_CHECK(cudaMemcpy(d_gt_base, h_gt_base.data(), blocks * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eq_base, h_eq_base.data(), blocks * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eq_accept, h_eq_accept.data(), blocks * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // ---- 收集 ----
    collect_kernel<<<blocks, threads>>>(input, N, threshold,
                                        d_gt_base, d_eq_base, d_eq_accept, output);
    CUDA_CHECK(cudaGetLastError());

    // ---- Bitonic Sort 降序输出 ----
    int m = 1;
    while (m < k) m <<= 1;

    CUDA_CHECK(cudaMalloc(&d_sort, m * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sort, output, k * sizeof(float), cudaMemcpyDeviceToDevice));

    if (m > k) {
        int fb = (m - k + threads - 1) / threads;
        if (fb < 1) fb = 1;
        fill_neg_inf_kernel<<<fb, threads>>>(d_sort + k, m - k);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int size = 2; size <= m; size <<= 1) {
        for (int j = size >> 1; j > 0; j >>= 1) {
            int sb = (m + threads - 1) / threads;
            if (sb < 1) sb = 1;
            bitonic_sort_step_desc<<<sb, threads>>>(d_sort, m, j, size);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 前 k 个已经是降序，直接拷回 output
    CUDA_CHECK(cudaMemcpy(output, d_sort, k * sizeof(float), cudaMemcpyDeviceToDevice));

    // ---- 清理 ----
    cudaFree(d_hist);
    cudaFree(d_block_gt);
    cudaFree(d_block_eq);
    cudaFree(d_gt_base);
    cudaFree(d_eq_base);
    cudaFree(d_eq_accept);
    cudaFree(d_sort);
}