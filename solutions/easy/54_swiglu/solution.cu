/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0406 ms; CUDA event: 0.0236 ms
*/

#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[halfN + idx];
        output[idx] = x1 / (1 + expf(-x1)) * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
