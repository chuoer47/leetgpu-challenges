/*
PASS: 9 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0349 ms; CUDA event: 0.0205 ms
*/

#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = x / (1 + expf(-x));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
