#include <cuda_runtime.h>
#define TILE 256

__global__ void histogram_kernel(
    const int* input,
    int* histogram,
    int N,
    int num_bins
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        int val = input[idx];
        atomicAdd(histogram + val, 1);
    }
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threads = TILE;
    int block = (N + threads - 1) / threads;
    histogram_kernel<<<block,threads>>>(input,histogram,N,num_bins);

    cudaDeviceSynchronize();
}   
