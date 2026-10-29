/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1765 ms; CUDA event: 0.1638 ms
*/

#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
