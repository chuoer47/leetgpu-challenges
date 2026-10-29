/*
读合并访问
PASS: 14 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.4413 ms; CUDA event: 0.4237 ms

写合并访问
PASS: 14 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.4383 ms; CUDA event: 0.4209 ms
*/

#include <cuda_runtime.h>

#define TILE 16
#define PAD 1

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float S[TILE][TILE + PAD]; 

    // 读合并访问
    int tiy = threadIdx.y;
    int tix = threadIdx.x;

    // 写合并访问
    // int tiy = threadIdx.x;
    // int tix = threadIdx.y;

    int row = blockIdx.y * TILE + tiy;
    int col = blockIdx.x * TILE + tix;
    if (row < rows && col < cols) {
        S[tix][tiy] = input[row * cols + col];
    } else {
        S[tix][tiy] = 0.0f;
    }
    __syncthreads();

    if (row < rows && col < cols) {
        output[col * rows + row] = S[tix][tiy];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
