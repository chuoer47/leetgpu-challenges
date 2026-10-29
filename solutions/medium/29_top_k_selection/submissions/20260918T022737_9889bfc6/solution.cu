#include <cuda_runtime.h>
#include <float.h>
#include <vector>

constexpr int TILE_SIZE = 256;

constexpr int ceil_div(int m,int n){
    return (m + n - 1) / n;
}

template <typename T>
__device__ void swap(T& a,T& b){
    T tmp = a;
    a = b;
    b = tmp;
}

__global__ void bitonic_sort_kernel(float* input,int len,int stride,int N){
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = a_idx ^ stride;

    if (a_idx >= N || b_idx >= N) return;

    if (b_idx > a_idx){
        if ((a_idx & len) ? (input[a_idx] > input[b_idx]):(input[a_idx] < input[b_idx])){
            swap(input[a_idx],input[b_idx]);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int n_pow2 = 1;
    while (n_pow2 < N){
        n_pow2 <<= 1;
    }

    float* padding_input = nullptr;
    cudaMalloc(&padding_input,sizeof(float) * n_pow2);

    std::vector<float> padding_input_cpu(n_pow2,-FLT_MAX);
    cudaMemcpy(padding_input,
        padding_input_cpu.data(),
        sizeof(float)*n_pow2,
        cudaMemcpyHostToDevice);
    
    cudaMemcpy(padding_input,
        input,
        sizeof(float) * N,
        cudaMemcpyHostToDevice);
    
    for (int len=2;len <= n_pow2;len <<= 1){
        for (int stride=(len >> 1);stride > 0;stride >>= 1){
            int grid = ceil_div(n_pow2,TILE_SIZE);
            bitonic_sort_kernel<<<grid,TILE_SIZE>>>(
                padding_input,len,stride,n_pow2
            );
        }
    }

    cudaMemcpy(output,
        padding_input,
        sizeof(float) * k,
        cudaMemcpyDeviceToDevice);

    cudaFree(padding_input);
}
