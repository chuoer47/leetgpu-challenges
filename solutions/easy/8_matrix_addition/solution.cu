/*
PASS: 13 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.2600 ms; CUDA event: 0.2427 ms
*/

#include <cuda_runtime.h>

#define TILE 256

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * TILE + threadIdx.x;
    if (idx < N * N)
        C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
