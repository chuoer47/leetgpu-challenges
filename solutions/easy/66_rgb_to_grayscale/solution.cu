/*
PASS: 16 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1153 ms; CUDA event: 0.0983 ms
*/

#include <cuda_runtime.h>
#define TILE 256

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (idx < N) {
        int start = 3 * idx;
        output[idx] = 0.299 * input[start] + 0.587 * input[start + 1] + 0.114 * input[start + 2];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = TILE;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
