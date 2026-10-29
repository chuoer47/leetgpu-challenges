/*
PASS: 16 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 112.7670 ms; CUDA event: 112.7505 ms
*/

#include <cuda_runtime.h>

#define TILE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    __shared__ float A_block[TILE][TILE];
    __shared__ float B_block[TILE][TILE];

    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int bix = blockIdx.x;
    const int biy = blockIdx.y;

    const int row = biy * TILE + tiy;
    const int col = bix * TILE + tix;

    float sum = 0.0;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        A_block[tiy][tix] = (row < M && (t * TILE + tix) < N) ? A[t * TILE + tix + row * N] : 0.0;
        B_block[tiy][tix] = (col < K && (t * TILE + tiy) < N) ? B[(t * TILE + tiy) * K + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += A_block[tiy][k] * B_block[k][tix];
        }

        __syncthreads(); // 如果这里不同步的话，可以会导致A_block/B_block被修改，导致累加出问题
    }

    if (row < M && col < K) {
        C[col + row * K] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
