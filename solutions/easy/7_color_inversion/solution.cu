/*
PASS: 9 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1929 ms; CUDA event: 0.1790 ms
*/

#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 4 * width * height) {
        image[idx] = 255 - image[idx];
        image[idx + 1] = 255 - image[idx + 1];
        image[idx + 2] = 255 - image[idx + 2];
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
