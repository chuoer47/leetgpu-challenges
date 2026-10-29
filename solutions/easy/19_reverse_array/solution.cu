/*
PASS: 13 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.2544 ms; CUDA event: 0.2406 ms
*/
#include <cuda_runtime.h>
#define TILE 256

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        float a = input[N - 1 - idx];
        float b = input[idx];
        input[N - 1 - idx] = b;
        input[idx] = a;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    if (N <= 1)
        return;
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
