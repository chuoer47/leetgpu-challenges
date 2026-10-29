/*
PASS: 10 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.4575 ms; CUDA event: 0.4444 ms
*/

#include <cuda_runtime.h>
#define TILE 256

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = (x <= 0) ? 0.01f * x :  x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
