/*
PASS: 12 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1617 ms; CUDA event: 0.1454 ms
*/

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32

struct __align__(8) MD {
    float m;
    float d;
};

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t error = (expr);                                                                \
        if (error != cudaSuccess) {                                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error));                                               \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

__device__ __forceinline__ MD md_identity() {
    return MD{-CUDART_INF_F, 0.0f};
}

__device__ __forceinline__ MD combine_md(MD a, MD b) {
    // d == 0 表示归约单位元，避免 -inf - (-inf)。
    if (a.d == 0.0f) {
        return b;
    }
    if (b.d == 0.0f) {
        return a;
    }

    // 也能处理两个相同的正负无穷值。
    if (a.m == b.m) {
        return MD{a.m, a.d + b.d};
    }

    if (a.m > b.m) {
        return MD{a.m, a.d + b.d * __expf(b.m - a.m)};
    }

    return MD{b.m, b.d + a.d * __expf(a.m - b.m)};
}

__device__ __forceinline__ MD warp_reduce_md(MD value) {
    constexpr unsigned mask = 0xffffffffu;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        MD other;
        other.m = __shfl_down_sync(mask, value.m, offset);
        other.d = __shfl_down_sync(mask, value.d, offset);
        value = combine_md(value, other);
    }

    return value;
}

template <int BLOCK_THREADS> __device__ __forceinline__ MD block_reduce_md(MD value) {
    static_assert(BLOCK_THREADS >= WARP_SIZE);
    static_assert(BLOCK_THREADS <= 1024);
    static_assert(BLOCK_THREADS % WARP_SIZE == 0);

    constexpr int WARP_COUNT = BLOCK_THREADS / WARP_SIZE;

    __shared__ MD warp_results[WARP_COUNT];

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    value = warp_reduce_md(value);

    if (lane_id == 0) {
        warp_results[warp_id] = value;
    }

    __syncthreads();

    MD block_value = md_identity();

    if (warp_id == 0) {
        if (lane_id < WARP_COUNT) {
            block_value = warp_results[lane_id];
        }

        // 剩余 lane 持有单位元，因此可以直接归约完整 warp。
        block_value = warp_reduce_md(block_value);
    }

    return block_value;
}

// 第一层：float 输入转换为 MD，并进行 block 级归约。
template <int BLOCK_THREADS>
__global__ void softmax_input_reduce_kernel(const float* input, MD* block_output, int count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;

    MD value = md_identity();

    if (index < static_cast<size_t>(count)) {
        value.m = input[index];
        value.d = 1.0f;
    }

    MD block_value = block_reduce_md<BLOCK_THREADS>(value);

    if (threadIdx.x == 0) {
        block_output[blockIdx.x] = block_value;
    }
}

// 后续层：MD 数组归约成更小的 MD 数组。
template <int BLOCK_THREADS>
__global__ void softmax_md_reduce_kernel(const MD* input, MD* block_output, int count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;

    MD value = md_identity();

    if (index < static_cast<size_t>(count)) {
        value = input[index];
    }

    MD block_value = block_reduce_md<BLOCK_THREADS>(value);

    if (threadIdx.x == 0) {
        block_output[blockIdx.x] = block_value;
    }
}

template <int BLOCK_THREADS>
__global__ void softmax_normalize_kernel(const float* input, float* output, const MD* global_md,
                                         int count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;

    if (index >= static_cast<size_t>(count)) {
        return;
    }

    const MD global = *global_md;
    const float x = input[index];
    const float inv_d = __fdividef(1.0f, global.d);

    // 对包含 +/- infinity 的输入提供定义明确的结果：
    // 最大值为 infinity 时，在所有相同 infinity 元素之间均分。
    if (isinf(global.m)) {
        output[index] = (x == global.m) ? inv_d : 0.0f;
    } else {
        output[index] = __expf(x - global.m) * inv_d;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    constexpr int THREADS = 256;

    if (N <= 0) {
        return;
    }

    const int first_block_count =
        static_cast<int>((static_cast<long long>(N) + THREADS - 1) / THREADS);

    MD* buffer_a = nullptr;
    MD* buffer_b = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer_a),
                          static_cast<size_t>(first_block_count) * sizeof(MD)));

    if (first_block_count > 1) {
        const int second_buffer_capacity = (first_block_count + THREADS - 1) / THREADS;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer_b),
                              static_cast<size_t>(second_buffer_capacity) * sizeof(MD)));
    }

    // Level 0: N 个 float -> first_block_count 个 MD。
    softmax_input_reduce_kernel<THREADS><<<first_block_count, THREADS>>>(input, buffer_a, N);

    CUDA_CHECK(cudaGetLastError());

    MD* current_input = buffer_a;
    MD* current_output = buffer_b;
    int current_count = first_block_count;

    // 递归归约：
    // first_block_count -> ceil(first_block_count / 256) -> ... -> 1
    while (current_count > 1) {
        const int next_count = (current_count + THREADS - 1) / THREADS;

        softmax_md_reduce_kernel<THREADS>
            <<<next_count, THREADS>>>(current_input, current_output, current_count);

        CUDA_CHECK(cudaGetLastError());

        MD* temporary = current_input;
        current_input = current_output;
        current_output = temporary;
        current_count = next_count;
    }

    // current_input 现在指向唯一的全局 MD。
    softmax_normalize_kernel<THREADS>
        <<<first_block_count, THREADS>>>(input, output, current_input, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(buffer_a));

    if (buffer_b != nullptr) {
        CUDA_CHECK(cudaFree(buffer_b));
    }
}