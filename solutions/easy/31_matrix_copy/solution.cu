/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1839 ms; CUDA event: 0.1669
*/

#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int vec4 = N * N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    if (idx < vec4) {
        B4[idx] = A4[idx];
    } else if (idx == vec4 && N * N % 4 != 0) {
        B[4*vec4] = A[4*vec4];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N / 4 + (N * N % 4 == 0 ? 0 : 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
