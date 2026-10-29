/*
PASS: 12 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 1.1149 ms; CUDA event: 1.1017 ms
*/

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int numWarps = blockDim.x >> 5;

    // Step 1: Warp-level reduction
    val = warpReduceMax(val);

    // Step 2: Write warp results
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // Step 3: First warp reduces all warp results
    if (warpId == 0) {
        float v = (lane < numWarps) ? shared[lane] : -FLT_MAX;
        v = warpReduceMax(v);
        if (lane == 0) {
            shared[0] = v;
        }
    }
    __syncthreads();

    return shared[0];
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int numWarps = blockDim.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < numWarps) ? shared[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0) {
            shared[0] = v;
        }
    }
    __syncthreads();

    return shared[0];
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;

    __shared__ float shared[8];

    // Step 1: Max
    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    float max_val = blockReduceMax(local_max, shared);

    // Step 2: Exp + Sum
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(input[i] - max_val);
        output[i] = val;
        local_sum += val;
    }

    float sum = blockReduceSum(local_sum, shared);

    // Step 3: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] /= sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0)
        return;

    constexpr int threadsPerBlock = 256;
    softmax_kernel<<<1, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}