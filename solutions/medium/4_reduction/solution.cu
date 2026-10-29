/*
PASS: 10 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0497 ms; CUDA event: 0.0369 ms
*/
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (idx < N)
        sum = input[idx];

    const int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    __shared__ float sum_s[WARP_NUM];

#pragma unroll
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, s);

    if (laneId == 0)
        sum_s[warpId] = sum;

    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        sum = (threadIdx.x < WARP_NUM) ? sum_s[threadIdx.x] : 0.0f;
#pragma unroll
        for (int s = WARP_NUM >> 1; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, s);
        if (threadIdx.x == 0)
            atomicAdd(output, sum);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
