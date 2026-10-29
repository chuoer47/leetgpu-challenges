/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0345 ms; CUDA event: 0.0215 ms
*/

#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(lo, fminf(input[idx], hi));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
