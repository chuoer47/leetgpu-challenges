/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0423 ms; CUDA event: 0.0285 ms
*/

#include <cuda_runtime.h>

#define rsqrt2 0.7071067811865475

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[halfN + idx];
        output[idx] = x1 * (0.5 * x2 * (1 + erff(x2 * rsqrt2)));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
