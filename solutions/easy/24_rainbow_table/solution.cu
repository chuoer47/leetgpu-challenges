/*
PASS: 9 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0716 ms; CUDA event: 0.0584 ms
*/

#include <cuda_runtime.h>
#define TILE 256

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    // 可以简单操作，先存下来，然后循环哈希，再存回去
    int tix = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        unsigned int v = static_cast<unsigned int>(input[idx]);
        for (int i = 0; i < R; ++i) {
            v = fnv1a_hash(v);
        }
        output[idx] = v;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
