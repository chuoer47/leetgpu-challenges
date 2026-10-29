/*
PASS: 15 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 1.3961 ms; CUDA event: 1.3834 ms
*/
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int output_size = input_size - kernel_size + 1;
    float sum = 0.0f;
    if (idx < output_size) {
        for (int i = 0; i < kernel_size; ++i) {
            sum += input[idx + i] * kernel[i];
        }
        output[idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
