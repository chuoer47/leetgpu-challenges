/*
PASS: 14 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.5971 ms; CUDA event: 0.5836 ms
*/
#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // GM input[row][col] -> Register -> GM output[col][row]
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < rows && col < cols)
        output[col * rows + row] = input[row * cols + col];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
