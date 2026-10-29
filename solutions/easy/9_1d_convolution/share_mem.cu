/*
PASS: 15 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 22.7420 ms; CUDA event: 22.7236 ms
*/

#include <cuda_runtime.h>

#define TILE 256
#define MAXK 2048

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    const int output_size = input_size - kernel_size + 1;
    __shared__ float K_[MAXK];
    int tix = threadIdx.x;
    int loop_ = (kernel_size + TILE - 1) / TILE;
    for (int i = 0; i < loop_; ++i) {
        if (tix + i * TILE < kernel_size) {
            K_[tix + i * TILE] = kernel[tix + i * TILE];
        }
    }
    __syncthreads();
    float sum = 0.0;
    int idx = blockIdx.x * blockDim.x + tix;
    for (int i = 0; i < kernel_size; i++) {
        sum += (idx + i < input_size) ? K_[i] * input[idx + i] : 0.0;
    }
    if (idx < output_size) {
        output[idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = TILE;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
