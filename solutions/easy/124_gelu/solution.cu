/*
PASS: 11 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.5280 ms; CUDA event: 0.5151 ms
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* X, float* Y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float tmp = 1.0f / sqrtf(2.0f);
    if (idx < N){
        float x = X[idx];
        Y[idx] = x * 0.5 * (1 + erff(x * tmp));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
