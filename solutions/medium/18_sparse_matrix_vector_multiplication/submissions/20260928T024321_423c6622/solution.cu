#include <cuda_runtime.h>
constexpr int TILE = 256;

__global__ void kernel(
    const float* A, const float* x, float* y, int M, int N
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M){
        return;
    }
    float sum = 0.0f;
    
    for (int i = 0;i < N;i++){
        float p = A[idx * N + i];
        if (p != 0.0f){
            sum += p * x[i];
        }
    }
    if (sum != 0.0f){
        y[idx] = sum;
    }

}

// A, x, y are device pointers
// A:[M,N] x:[N:1] y:[M,1]
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    // 先写一版最简单的
    cudaMemset(y, 0, M * sizeof(int));
    int block = (M + TILE - 1) / TILE;
    kernel<<<block,TILE>>>(A,x,y,M,N);
    cudaDeviceSynchronize();
}
