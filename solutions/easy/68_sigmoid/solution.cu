/*
PASS: 15 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.4598 ms; CUDA event: 0.4455 ms
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        Y[idx] = 1 / (1 + expf(-X[idx]));
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
