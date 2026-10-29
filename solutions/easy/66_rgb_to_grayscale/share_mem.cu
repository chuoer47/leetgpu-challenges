/*
PASS: 16 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1121 ms; CUDA event: 0.0983 ms
*/

#include <cuda_runtime.h>
#define TILE 256

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x;
    int start = 3 * idx;
    int tix = threadIdx.x;
    int N = width * height;
    __shared__ float A[TILE * 3];
    // 需要取 [start,start + 3*TILE] 的数字
    // 为了保证少取，同时让每个线程都取
    for (int i = tix; start + i < 3 * N && i < 3 * TILE; i += TILE) {
        A[i] = input[start + i];
    }
    __syncthreads();
    if (idx + tix < N) {
        output[idx + tix] = 0.299 * A[3 * tix] + 0.587 * A[3 * tix + 1] + 0.114 * A[3 * tix + 2];
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
